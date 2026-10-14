"""Require a versioned tag on tested, protected main before release work begins."""

from __future__ import annotations

import argparse
import json
import os
import re
import tomllib
import urllib.request
from pathlib import Path

REPOSITORY = "Synthefy/nori-rel"


def read_github_repository_json(path: str) -> dict:
    """Read GitHub release evidence with the workflow's read-only token."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPOSITORY}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    """Verify the selected release and emit immutable identifiers for later jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--target", choices=("testpypi", "pypi"), required=True)
    parser.add_argument("--candidate-run", default="")
    args = parser.parse_args()
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError("only Synthefy/nori-rel can publish this distribution")
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("dispatch releases from protected main only")
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", args.tag):
        raise ValueError("release tag must be vX.Y.Z, optionally followed by a/b/rcN")
    tag = read_github_repository_json(f"git/ref/tags/{args.tag}")["object"]
    for _ in range(5):
        if tag["type"] != "tag":
            break
        tag = read_github_repository_json(f"git/tags/{tag['sha']}")["object"]
    if tag["type"] != "commit":
        raise ValueError("release tag must resolve to a commit")
    tag_commit = tag["sha"]
    main_branch = read_github_repository_json("branches/main")
    if not main_branch.get("protected"):
        raise ValueError("main branch protection must be enabled before publishing")
    if tag_commit != main_branch["commit"]["sha"]:
        raise ValueError("release tag must point to the current protected main commit")
    if tag_commit != os.environ.get("GITHUB_SHA"):
        raise ValueError("main moved after dispatch; dispatch a fresh release run")
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if args.tag != f"v{version}":
        raise ValueError("release tag and project version differ")
    runs = read_github_repository_json(f"commits/{tag_commit}/check-runs?per_page=100")[
        "check_runs"
    ]
    readiness = sorted(
        (
            run
            for run in runs
            if run["name"] == "release-readiness"
            and run.get("app", {}).get("slug") == "github-actions"
        ),
        key=lambda run: run["id"],
        reverse=True,
    )
    if not readiness or readiness[0]["conclusion"] != "success":
        raise ValueError("the current commit must pass the release-readiness check")
    run_url = readiness[0]["details_url"]
    match = re.search(r"/actions/runs/(\d+)", run_url)
    if match is None:
        raise ValueError("cannot identify the readiness workflow run")
    run = read_github_repository_json(f"actions/runs/{match[1]}")
    if (
        run["path"] != ".github/workflows/ci.yml"
        or run["event"] != "push"
        or run["head_branch"] != "main"
        or run["head_sha"] != tag_commit
        or run["conclusion"] != "success"
    ):
        raise ValueError("readiness must come from successful CI on the main push")
    if args.target == "pypi":
        if not re.fullmatch(r"[1-9]\d*", args.candidate_run):
            raise ValueError("PyPI requires a successful TestPyPI candidate run ID")
        candidate = read_github_repository_json(f"actions/runs/{args.candidate_run}")
        if (
            candidate["path"] != ".github/workflows/publish.yml"
            or candidate["event"] != "workflow_dispatch"
            or candidate["head_branch"] != "main"
            or candidate["head_sha"] != tag_commit
            or candidate["conclusion"] != "success"
            or candidate.get("repository", {}).get("full_name") != REPOSITORY
            or candidate.get("head_repository", {}).get("full_name") != REPOSITORY
        ):
            raise ValueError(
                "candidate must be a successful release run at this commit"
            )
    elif args.candidate_run:
        raise ValueError("TestPyPI builds a new candidate; omit candidate_run")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"commit={tag_commit}\nversion={version}\n")
    print(f"Release {args.tag} verified at {tag_commit}")


if __name__ == "__main__":
    main()
