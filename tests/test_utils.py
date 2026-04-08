from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from mosaic import utils


class UtilsTests(unittest.TestCase):
    def test_embed_texts_suppresses_third_party_output(self) -> None:
        class FakeModel:
            def __init__(self, model_name: str, local_files_only: bool = True) -> None:
                print(f'loading {model_name} local={local_files_only}')

            def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False):
                print(f'encoding {len(texts)} texts')
                assert show_progress_bar is False
                return [[1.0, 0.0] for _ in texts]

        utils._SENTENCE_MODEL_CACHE.clear()
        utils._SENTENCE_MODEL_FAILURES.clear()
        with patch.object(utils, 'SentenceTransformer', FakeModel):
            captured = io.StringIO()
            with redirect_stdout(captured), redirect_stderr(captured):
                rows = utils.embed_texts(['alpha', 'beta'], model_name='fake-model', dimensions=2)
        self.assertEqual(captured.getvalue(), '')
        self.assertEqual(rows, [[1.0, 0.0], [1.0, 0.0]])



    def test_generic_roles_default_to_exact_match_plus_public(self) -> None:
        self.assertEqual(utils.resolve_user_roles(['finance']), {'finance', 'public'})
        self.assertTrue(utils.roles_permit(['public'], ['finance']))
        self.assertTrue(utils.roles_permit(['finance'], ['finance']))
        self.assertFalse(utils.roles_permit(['legal'], ['finance']))

    def test_custom_role_inheritance_and_wildcard_access(self) -> None:
        inheritance = {
            'manager': ['member', 'manager'],
            'admin': ['member', 'manager', 'admin'],
        }
        self.assertEqual(utils.resolve_user_roles(['manager'], role_inheritance=inheritance), {'public', 'member', 'manager'})
        self.assertTrue(utils.roles_permit(['member'], ['manager'], role_inheritance=inheritance))
        self.assertTrue(utils.roles_permit(['restricted'], ['*']))

if __name__ == '__main__':
    unittest.main()
