from pathlib import Path
from .build import build

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="cva.attacklab")
    ap.add_argument("--out", default="artifacts/corpus")
    a = ap.parse_args()
    build(Path(a.out))
