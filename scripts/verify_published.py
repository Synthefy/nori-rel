"""Download index release files and verify they match the approved candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def verify_published_distributions(
    index: str, manifest: dict, destination: Path
) -> None:
    """Verify the index metadata and each downloaded distribution's actual bytes."""
    hostname = "pypi.org" if index == "pypi" else "test.pypi.org"
    version = urllib.parse.quote(manifest["version"], safe="")
    url = f"https://{hostname}/pypi/nori-rel/{version}/json"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                release = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code != 404 or attempt == 4:
                raise
            time.sleep(10)
    published = {entry["filename"]: entry for entry in release["urls"]}
    if set(published) != set(manifest["files"]):
        raise ValueError("index release files differ from the approved candidate")
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        entry = published[name]
        if Path(name).name != name or entry["digests"]["sha256"] != expected["sha256"]:
            raise ValueError(f"index metadata does not match the candidate: {name}")
        parsed = urllib.parse.urlsplit(entry["url"])
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(
            ".pythonhosted.org"
        ):
            raise ValueError("index returned an unexpected file host")
        with urllib.request.urlopen(entry["url"], timeout=60) as response:
            contents = response.read()
        if (
            hashlib.sha256(contents).hexdigest() != expected["sha256"]
            or len(contents) != expected["size"]
        ):
            raise ValueError(f"downloaded file differs from the candidate: {name}")
        (destination / name).write_bytes(contents)
    print(f"Verified every {index} release file against the approved candidate")


def main() -> None:
    """Read the immutable candidate manifest and verify a package index release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=("pypi", "testpypi"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_published_distributions(
        args.index, json.loads(args.manifest.read_text()), args.output
    )


if __name__ == "__main__":
    main()
