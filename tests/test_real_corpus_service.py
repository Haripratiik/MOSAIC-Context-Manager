from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from mosaic.ingestor import ingest_directory
from mosaic.service import MosaicService, QueryRequest


class RealCorpusServiceTests(unittest.TestCase):
    def test_uncalibrated_real_corpus_query_can_succeed(self) -> None:
        root = Path('.mosaic_test') / f"real_corpus_service_{uuid4().hex}"
        docs_dir = root / 'docs'
        docs_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / 'deliverables.txt').write_text(
            'The remaining manual-only deliverables are the architecture diagram PDF and the demo screencast recording.',
            encoding='utf-8',
        )

        index_path = root / 'index.json'
        ingest_directory(
            docs_dir,
            output_path=index_path,
            chunk_size=64,
            overlap=8,
            embedding_model='hash',
            vector_store='json',
        )
        service = MosaicService(index_path)
        response = service.query(
            QueryRequest(
                query='Which deliverables are manual-only?',
                requested_roles=['public'],
                strategy='mosaic_no_ledger',
                token_budget=256,
            )
        )

        self.assertEqual(response.status, 'success')
        self.assertGreaterEqual(len(response.selected_chunk_titles), 1)


if __name__ == '__main__':
    unittest.main()
