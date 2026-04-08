from __future__ import annotations

import unittest

from mosaic.ledger import ContextLedger
from mosaic.service import QueryRequest
from tests.helpers import build_fixture, pick_query


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = build_fixture('service')
        cls.service = fixture['service']
        cls.suite = fixture['suite']

    def test_query_returns_all_four_statuses(self) -> None:
        success_item = pick_query(self.suite, category='redundancy_trap')
        true_unknown_item = pick_query(self.suite, category='failure_classification', ground_truth_type='TRUE_UNKNOWN')
        permission_item = pick_query(self.suite, category='failure_classification', ground_truth_type='PERMISSION_GAP')
        retrieval_item = pick_query(self.suite, category='failure_classification', ground_truth_type='RETRIEVAL_FAILURE')

        success = self.service.query(QueryRequest(query=success_item['query'], requested_roles=list(success_item['user_roles']), strategy='mosaic_full', token_budget=900))
        true_unknown = self.service.query(QueryRequest(query=true_unknown_item['query'], requested_roles=list(true_unknown_item['user_roles']), strategy='mosaic_full', token_budget=900))
        permission_gap = self.service.query(QueryRequest(query=permission_item['query'], requested_roles=list(permission_item['user_roles']), strategy='mosaic_full', token_budget=900))
        retrieval_failure = self.service.query(QueryRequest(query=retrieval_item['query'], requested_roles=list(retrieval_item['user_roles']), strategy='mosaic_full', token_budget=900))

        self.assertEqual(success.status, 'success')
        self.assertEqual(true_unknown.status, 'true_unknown')
        self.assertEqual(permission_gap.status, 'permission_gap')
        self.assertEqual(retrieval_failure.status, 'retrieval_failure')
        self.assertTrue(success.audit_id)
        self.assertTrue(permission_gap.audit_id)

    def test_audit_trace_and_conversation_export_include_hydrated_chunks(self) -> None:
        item = pick_query(self.suite, category='redundancy_trap')
        response = self.service.query(QueryRequest(query=item['query'], requested_roles=list(item['user_roles']), strategy='mosaic_full', token_budget=900, conversation_id='svc-conversation', turn=1))
        trace = self.service.get_audit_trace(response.audit_id, include_text=True)
        export = self.service.export_audit(conversation_id='svc-conversation', include_text=True)

        self.assertEqual(trace['audit_id'], response.audit_id)
        self.assertGreater(len(trace['candidates']), 0)
        self.assertGreaterEqual(len(trace['selected_chunks']), 1)
        self.assertIn('text', trace['selected_chunks'][0])
        self.assertEqual(export['mode'], 'conversation')
        self.assertEqual(export['conversation_id'], 'svc-conversation')
        self.assertGreaterEqual(len(export['requests']), 1)

    def test_sqlite_ledger_round_trip_preserves_embeddings(self) -> None:
        item = pick_query(self.suite, category='redundancy_trap')
        response = self.service.query(
            QueryRequest(
                query=item['query'],
                requested_roles=list(item['user_roles']),
                strategy='mosaic_full',
                token_budget=900,
                conversation_id='svc-ledger-roundtrip',
                turn=1,
            )
        )
        self.assertTrue(response.conversation_id)
        ledger = self.service.ledger_store.load('svc-ledger-roundtrip', 900, ContextLedger)
        self.assertGreater(len(ledger.selected_embeddings), 0)
        self.assertEqual(len(ledger.selected_embeddings), len(ledger.selected_chunk_ids))

    def test_non_ledger_query_does_not_allocate_conversation(self) -> None:
        item = pick_query(self.suite, category='redundancy_trap')
        response = self.service.query(
            QueryRequest(
                query=item['query'],
                requested_roles=list(item['user_roles']),
                strategy='mosaic_no_ledger',
                token_budget=700,
            )
        )
        trace = self.service.get_audit_trace(response.audit_id, include_text=False)

        self.assertIsNone(response.conversation_id)
        self.assertIsNone(trace['conversation'])

    def test_invalid_request_inputs_raise_value_error(self) -> None:
        item = pick_query(self.suite, category='redundancy_trap')
        with self.assertRaises(ValueError):
            self.service.query(QueryRequest(query='   ', requested_roles=list(item['user_roles']), strategy='mosaic_full', token_budget=900))
        with self.assertRaises(ValueError):
            self.service.query(QueryRequest(query=item['query'], requested_roles=list(item['user_roles']), strategy='mosaic_full', token_budget=0))
        with self.assertRaises(ValueError):
            self.service.query(QueryRequest(query=item['query'], requested_roles=list(item['user_roles']), strategy='mosaic_full', token_budget=900, candidate_k=0))

    def test_permission_trace_and_non_selection_factors_are_structured(self) -> None:
        item = pick_query(self.suite, category='failure_classification', ground_truth_type='PERMISSION_GAP')
        response = self.service.query(QueryRequest(query=item['query'], requested_roles=list(item['user_roles']), strategy='mosaic_full', token_budget=900))
        trace = self.service.get_audit_trace(response.audit_id, include_text=False)

        self.assertEqual(trace['outcome']['required_role'], response.required_role)
        denied = [row for row in trace['candidates'] if not row['permitted']]
        self.assertTrue(denied)
        self.assertTrue(any('not_permitted' in row['factors'] for row in denied))
        for row in trace['candidates']:
            self.assertIsInstance(row['factors'], list)


if __name__ == '__main__':
    unittest.main()
