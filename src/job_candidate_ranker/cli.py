import argparse
import json
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Train and score the candidate ranking model.")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(run(args.data_dir, args.output_dir), indent=2))
