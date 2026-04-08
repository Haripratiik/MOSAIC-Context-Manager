from __future__ import annotations

import json
import unittest
from pathlib import Path
from uuid import uuid4

from mosaic.ingestor import ingest_directory


class IngestManifestTests(unittest.TestCase):
    def test_manifest_and_sidecar_metadata_are_ingested(self) -> None:
        root = Path('.mosaic_test') / f"ingest_manifest_{uuid4().hex}"
        docs_dir = root / 'docs'
        docs_dir.mkdir(parents=True, exist_ok=True)

        (docs_dir / 'policy.txt').write_text('Sensitive access policy for the control plane.', encoding='utf-8')
        (docs_dir / 'policy.txt.meta.json').write_text(
            json.dumps(
                {
                    'roles': ['compliance'],
                    'doc_type': 'policy',
                    'metadata': {'region': 'us'},
                }
            ),
            encoding='utf-8',
        )
        (docs_dir / 'notes.md').write_text('Analyst onboarding notes for the project.', encoding='utf-8')

        manifest_path = root / 'manifest.json'
        manifest_path.write_text(
            json.dumps(
                {
                    'source_root': 'docs',
                    'documents': [
                        {
                            'path': 'policy.txt',
                            'document_id': 'policy-doc',
                            'title': 'Control Plane Policy',
                        },
                        {
                            'path': 'notes.md',
                            'document_id': 'notes-doc',
                            'roles': ['analyst'],
                            'metadata': {'team': 'research'},
                        },
                    ],
                }
            ),
            encoding='utf-8',
        )

        index = ingest_directory(
            None,
            output_path=root / 'index.json',
            chunk_size=64,
            overlap=8,
            embedding_model='hash',
            vector_store='json',
            manifest_path=manifest_path,
        )

        self.assertEqual(index.config['corpus_source'], 'manifest')
        self.assertEqual(index.config['manifest_path'], str(manifest_path))
        policy_chunk = next(chunk for chunk in index.chunks if chunk.document_id == 'policy-doc')
        notes_chunk = next(chunk for chunk in index.chunks if chunk.document_id == 'notes-doc')
        self.assertEqual(policy_chunk.roles, ['compliance'])
        self.assertEqual(policy_chunk.metadata['doc_type'], 'policy')
        self.assertEqual(policy_chunk.metadata['region'], 'us')
        self.assertEqual(notes_chunk.roles, ['analyst'])
        self.assertEqual(notes_chunk.metadata['team'], 'research')

    def test_directory_ingest_respects_sidecar_metadata(self) -> None:
        root = Path('.mosaic_test') / f"ingest_sidecar_{uuid4().hex}"
        docs_dir = root / 'docs'
        docs_dir.mkdir(parents=True, exist_ok=True)

        (docs_dir / 'memo.txt').write_text('Escalation memo for a regulated workflow.', encoding='utf-8')
        (docs_dir / 'memo.txt.meta.json').write_text(
            json.dumps(
                {
                    'document_id': 'memo-doc',
                    'roles': ['executive'],
                    'title': 'Escalation Memo',
                    'metadata': {'desk': 'prime'},
                }
            ),
            encoding='utf-8',
        )

        index = ingest_directory(
            docs_dir,
            output_path=root / 'index.json',
            chunk_size=64,
            overlap=8,
            embedding_model='hash',
            vector_store='json',
        )

        memo_chunk = next(chunk for chunk in index.chunks if chunk.document_id == 'memo-doc')
        self.assertEqual(memo_chunk.roles, ['executive'])
        self.assertEqual(memo_chunk.metadata['title'], 'Escalation Memo')
        self.assertEqual(memo_chunk.metadata['desk'], 'prime')


if __name__ == '__main__':
    unittest.main()
