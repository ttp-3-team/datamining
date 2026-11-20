"""Freeze the train-only choice, then evaluate it once on the temporal test set."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from final_test_evaluation import evaluate_once, freeze_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true",
                        help="Lock the train-only-selected strategy without opening test files.")
    args = parser.parse_args()
    result = freeze_protocol(ROOT) if args.freeze_only else evaluate_once(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
