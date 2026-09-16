"""Guards on the things that tie the repository together rather than on any one module.

Three kinds of rot this catches, all of which have happened at least once here:

* a Makefile target named in the README, in CI or in a shell script that does not exist,
  so the command fails only when someone finally runs it;
* a documentation link to a page that was renamed or never written, which GitHub renders as
  a dead link rather than as an error;
* a performance number typed into a markdown page, which the results policy in
  ``CONTRIBUTING.md`` ("No numbers without a results JSON") forbids outside tables
  rendered from ``results/*.json``;
* a reference to a file that is not in the repository, which reads as a dangling pointer
  to whoever clones it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

#: Every target the README and CONTRIBUTING name, plus the ones the scripts call.
REQUIRED_TARGETS: tuple[str, ...] = (
    "setup",
    "lint",
    "typecheck",
    "test",
    "test-slow",
    "test-gpu",
    "k8s-lint",
    "bench",
    "bench-h100",
    "results",
    "docker-build",
)


def makefile_targets() -> set[str]:
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    return {match.group(1) for match in re.finditer(r"^([a-zA-Z0-9_-]+):", text, re.MULTILINE)}


@pytest.mark.parametrize("target", REQUIRED_TARGETS)
def test_makefile_has_the_target(target: str) -> None:
    assert target in makefile_targets()


def test_every_required_target_is_phony() -> None:
    """A real file called `test` in the tree would otherwise silently disable `make test`."""
    # Join line continuations first: the .PHONY list is wrapped across several lines.
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8").replace("\\\n", " ")
    phony = set(" ".join(re.findall(r"^\.PHONY:(.*)$", text, re.MULTILINE)).split())
    missing = [target for target in REQUIRED_TARGETS if target not in phony]
    assert not missing, f"not declared .PHONY: {missing}"


def test_run_remote_calls_a_target_that_exists() -> None:
    """`scripts/vastai/run_remote.sh` runs `make bench` on the instance."""
    text = (REPO_ROOT / "scripts" / "vastai" / "run_remote.sh").read_text(encoding="utf-8")
    assert "make bench" in text
    assert "bench" in makefile_targets()


def markdown_files() -> list[Path]:
    return sorted([*DOCS.rglob("*.md"), REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md"])


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.name))
def test_relative_links_resolve(path: Path) -> None:
    """Every relative markdown link points at a file that exists.

    Anchors are stripped rather than checked: a missing heading is a cosmetic problem, a
    missing file is a broken page.
    """
    text = path.read_text(encoding="utf-8")
    broken: list[str] = []
    for match in re.finditer(r"\[[^\]]*\]\(([^)\s]+)\)", text):
        target = match.group(1)
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        if not resolved.exists():
            broken.append(target)
    assert not broken, f"{path.relative_to(REPO_ROOT)} links to missing files: {broken}"


def test_docs_index_links_every_page() -> None:
    """A page nobody links to is a page nobody reads."""
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    pages = {path.name for path in DOCS.glob("*.md")} - {"index.md"}
    missing = sorted(page for page in pages if page not in index)
    assert not missing, f"docs/index.md does not link: {missing}"


def test_every_adr_is_listed() -> None:
    adr_index = (DOCS / "adr" / "README.md").read_text(encoding="utf-8")
    records = {path.name for path in (DOCS / "adr").glob("ADR-*.md")}
    assert len(records) >= 6
    missing = sorted(name for name in records if name not in adr_index)
    assert not missing, f"docs/adr/README.md does not list: {missing}"


#: Markdown that is generated from result files, and is therefore allowed to hold numbers.
GENERATED_PAGES = {"results.md"}

#: Units a performance claim would be written in. Deliberately narrow: this is a tripwire
#: for "3.1x faster" and "p95 of 42 ms", not a general number detector -- block sizes,
#: percentages of VRAM, port numbers and version strings are all legitimate.
PERFORMANCE_CLAIM = re.compile(
    r"\d+(?:\.\d+)?\s*(?:x\s+(?:faster|higher|lower|speedup|throughput)"
    r"|tokens?/s|tok/s|requests?/s|ms\s+(?:p50|p95|p99)"
    r"|(?:p50|p95|p99)\s+of)",
    re.IGNORECASE,
)


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.name))
def test_no_hand_written_performance_numbers(path: Path) -> None:
    if path.name in GENERATED_PAGES:
        return
    text = path.read_text(encoding="utf-8")
    hits = [match.group(0) for match in PERFORMANCE_CLAIM.finditer(text)]
    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} states performance numbers {hits}; "
        "numbers belong in results/*.json and the pages rendered from them"
    )


#: Files whose paths are named in prose all over the repository and must therefore exist.
#: ``PLAN.md`` and ``specs/`` were the working notes this repository was developed against;
#: they were never part of it, so a citation of one is a pointer a reader cannot follow.
DANGLING_REFERENCE = re.compile(r"PLAN\.md|\bspecs/", re.IGNORECASE)

#: An absolute path under a user's home directory only ever means one developer's checkout.
ABSOLUTE_HOME_PATH = re.compile(r"/(?:home|Users)/[a-z][a-z0-9._-]*/", re.IGNORECASE)

#: Directories whose contents are data or generated, not prose anyone reads for guidance.
UNSCANNED_DIRS = ("results/", ".git/")

#: Extensions that hold text a human wrote. Everything else (weights, images, lock files)
#: is skipped: a match inside a dependency name is noise, not a citation.
SCANNED_SUFFIXES = (".md", ".py", ".yaml", ".yml", ".sh", ".toml", ".cfg", ".txt", ".tpl")


def tracked_text_files() -> list[Path]:
    """Every tracked file that carries prose, as absolute paths."""
    import subprocess

    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:  # not a git checkout (an sdist, say)
        pytest.skip("not a git checkout")
    names = [name for name in listing.stdout.split("\0") if name]
    return [
        REPO_ROOT / name
        for name in names
        if name.endswith(SCANNED_SUFFIXES)
        and not name.startswith(UNSCANNED_DIRS)
        and name != "uv.lock"
    ]


def test_no_file_cites_a_document_that_is_not_in_the_repository() -> None:
    """A citation a reader cannot follow is worse than no citation.

    The policies those notes held now live in ``CONTRIBUTING.md``; cite that instead.
    """
    offenders: list[str] = []
    for path in tracked_text_files():
        if path.name == "test_repo_layout.py":
            continue  # the patterns themselves
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if DANGLING_REFERENCE.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, "references to files outside the repository:\n" + "\n".join(offenders)


def test_no_file_hard_codes_a_path_under_someones_home_directory() -> None:
    offenders: list[str] = []
    for path in tracked_text_files():
        if path.name == "test_repo_layout.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if ABSOLUTE_HOME_PATH.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not offenders, "absolute paths under a home directory:\n" + "\n".join(offenders)
