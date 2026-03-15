#!/usr/bin/env python3
import os
import subprocess
import sys
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path

TAG_PREFIX = "openswe-python"
APT_MIRROR = "deb.debian.org"  # your debian mirror
PYPI_MIRROR = "https://pypi.org/simple/"  # your pypi mirror
PYTHON_VERSIONS = ["2.7"] + [f"3.{i}" for i in range(5, 15)]  # 3.5 to 3.14
DOCKERFILE_TEMPLATE = f"""
FROM continuumio/miniconda3:25.3.1-1
RUN sed -i 's|deb.debian.org|{APT_MIRROR}|g' /etc/apt/sources.list.d/debian.sources && \\
 apt update && \\
 rm -rf /var/lib/apt/lists/*
RUN conda create -n testbed python={{python_version}} -y; \\
 echo "conda activate testbed" >> ~/.bashrc; \\
 conda activate testbed; \\
 pip config set global.index-url {PYPI_MIRROR};
"""


def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def main():
    if not shutil.which("docker"):
        print("❌ Docker is not installed or not in PATH.")
        sys.exit(1)

    with TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        basedir = Path(tempdir)
        for version in PYTHON_VERSIONS:
            try:
                tag = f"{TAG_PREFIX}-{version}"
                dockerfile_path = (
                    basedir / f"Dockerfile.python{version.replace('.', '_')}"
                )
                dockerfile_content = DOCKERFILE_TEMPLATE.format(python_version=version)
                with open(dockerfile_path, "w") as f:
                    f.write(dockerfile_content)

                try:
                    run_cmd(f"docker build -t {tag} -f {dockerfile_path} .")
                    # run_cmd(f"docker save -o /data/share/builder/scripts/images/{tag}.tar {tag}")
                    # run_cmd(f"docker run -d --name {tag} {tag} sleep infinity")
                    # run_cmd(f"docker start {tag}")
                    print(f"✅ Successfully built and tested {tag}")
                finally:
                    dockerfile_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"❌ Failed to build Python {version}: {e}")

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
