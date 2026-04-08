from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mosaic.corpus_builder import generate_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic MOSAIC corpus")
    parser.add_argument("--output-dir", default=str(ROOT / "corpus" / "documents"))
    parser.add_argument("--catalog", default=str(ROOT / "corpus" / "catalog.json"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    catalog = generate_corpus(args.output_dir, catalog_path=args.catalog, clean=args.clean)
    print(f"Generated {len(catalog)} documents in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
