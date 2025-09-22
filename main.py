# main.py
import argparse
import sys
from pathlib import Path

# 1) Make sure "src/" is on the import path when running: python main.py ...
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def main():
    parser = argparse.ArgumentParser(description="EmoVision - unified entry point")
    parser.add_argument("mode", choices=["collect", "train", "infer"], help="Which stage to run")
    args, rest = parser.parse_known_args()

    if args.mode == "collect":
        # ✅ IMPORT BACK IN
        from emovision.data import collector
        sys.argv = [sys.argv[0]] + rest
        collector.main()

    elif args.mode == "train":
        from emovision import train
        sys.argv = [sys.argv[0]] + rest
        train.main()

    elif args.mode == "infer":
        from emovision import infer
        sys.argv = [sys.argv[0]] + rest
        infer.main()

if __name__ == "__main__":
    main()
