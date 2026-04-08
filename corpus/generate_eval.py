from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mosaic.corpus_builder import generate_eval_suite
from mosaic.utils import load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the MOSAIC evaluation suite")
    parser.add_argument("--catalog", default=str(ROOT / "corpus" / "catalog.json"))
    parser.add_argument("--output", default=str(ROOT / "corpus" / "eval_suite.json"))
    args = parser.parse_args()

    catalog = load_json(args.catalog)
    suite = generate_eval_suite(catalog, output_path=args.output)
    print(f"Generated {suite["metadata"]["total_queries"]} evaluation items in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
