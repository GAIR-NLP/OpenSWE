"""Minimal logic to build an instance image."""

import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import prepare_baseimg


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _repo_url(repo: str) -> str:
    if repo.startswith(("http://", "https://", "git@")):
        return repo
    return f"https://github.com/{repo}.git"


def build_image(instance: dict) -> str:
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        repo_dir = tmpdir_path / "repo"
        dockerfile_path = tmpdir_path / "Dockerfile"
        dockerfile_path.write_text(instance["Dockerfile"], encoding="utf-8")

        _run(["git", "clone", _repo_url(instance["repo"]), str(repo_dir)])
        _run(["git", "checkout", instance["base_commit"]], cwd=repo_dir)
        _run(["docker", "build", "-t", instance["image_name"], "."], cwd=repo_dir)

    return instance["image_name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build images from a JSONL dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset JSONL file.")
    args = parser.parse_args()

    prepare_baseimg.main()
    with args.dataset_path.open(encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            build_image(sample)


if __name__ == "__main__":
    main()
