from pathlib import Path


def check_dir(directory: Path):
    if not directory.exists():
        directory.mkdir(parents=True)
