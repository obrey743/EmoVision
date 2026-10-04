# main.py
import argparse
import importlib
import sys
from pathlib import Path

# Make sure "src/" is on the import path when running: python main.py ...
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

MODULES = {
    "collect": "emovision.data.collector",
    "train": "emovision.train",
    "infer": "emovision.infer",
}


def main():
    parser = argparse.ArgumentParser(
        description="EmoVision - unified entry point",
        epilog="Run 'python main.py <mode> -h' for the options of each mode.",
    )
    parser.add_argument("mode", choices=MODULES, help="Which stage to run")
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()

    module = importlib.import_module(MODULES[args.mode])
    sys.argv = [f"{sys.argv[0]} {args.mode}"] + args.args
    module.main()


if __name__ == "__main__":
    main()
