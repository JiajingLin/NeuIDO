from pathlib import Path
import sys
import shutil


def get_package_root() -> Path:
    return Path(__file__).resolve().parent


def mkdir(path: Path, resume=False, overwrite=False, verbose: bool = False) -> tuple[bool, bool]:
    """
        resume: 如果为 True，表示如果目录存在则“继续”使用它，不删除
        overwrite: 如果为 True，则删除已有目录并重新创建
    """

    while True:
        if overwrite:
            if path.is_dir() and verbose:
                print(f'overwriting directory ({path})')
            shutil.rmtree(path, ignore_errors=True)
            path.mkdir(parents=True, exist_ok=True)
            return resume, overwrite
        if resume:
            if verbose:
                print(f'resuming directory ({path})')
            path.mkdir(parents=True, exist_ok=True)
            return resume, overwrite
        if path.exists():
            feedback = input(f'target directory ({path}) already exists, overwrite? [Y/r/n] ')
            ret = feedback.casefold()
        else:
            ret = 'y'
        if ret == 'n':
            sys.exit(0)
        elif ret == 'r':
            resume = True
        elif ret == 'y':
            overwrite = True


def get_root(path: str | Path, name: str = '.root') -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root
