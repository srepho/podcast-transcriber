from contextlib import closing
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).parents[1] / 'scripts/pilot_dataset.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.json'
        self.source.write_text(json.dumps({
            'guid': 'https://secret.invalid/PRIVATE_TOKEN', 'feed': 'demo', 'model': 'base.en',
            'published': '2024-03-01T00:00:00Z', 'audio_url': 'https://secret.invalid/PRIVATE_TOKEN',
            'segments': [{'start': 0.0, 'end': 10.0,
                          'text': 'Alex Example could miss games after an ankle injury.',
                          'raw_text': 'Alex Exampel could miss games after an ankle injury.'},
                         {'start': 10.0, 'end': 15.0, 'text': 'The team discussed the next game.'}]}))
        with closing(sqlite3.connect(self.root / 'podcast.db')) as db, db:
            db.execute('CREATE TABLE episodes (guid TEXT, feed_name TEXT, title TEXT, published TEXT, status TEXT, transcript_path TEXT)')
            db.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)',
                       ('https://secret.invalid/PRIVATE_TOKEN', 'demo', 'Daily One', '2024-03-01T00:00:00Z', 'transcribed', str(self.source)))
        self.bundle = self.root / 'bundle'
        with patch.object(pilot, 'now', return_value='2024-03-03T00:00:00+00:00'):
            pilot.build(self.root, 'demo', 'Daily', 5, self.bundle)

    def reviewed(self, status='accepted'):
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviewer'] = 'test-reviewer'
        review['reviews'][0].update(status=status, claim_text='Alex Example might miss games.',
            certainty='speculation', temporal_scope='current', entities=[{'type': 'player', 'name': 'Alex Example', 'model_entity_id': 'test:123'}],
            valid_until='2024-03-05T00:00:00Z', notes='Synthetic evidence verified; conditional, not confirmed.')
        pilot.write_json(review_path, review)
        dataset = self.root / 'finalized'
        with patch.object(pilot, 'now', return_value='2024-03-04T00:00:00+00:00'):
            pilot.finalize(self.bundle, review_path, dataset)
        return dataset

    def test_legacy_unknown_times_raw_text_and_no_private_urls(self):
        ep = pilot.read_rows(self.bundle / 'episodes.jsonl')[0]
        self.assertIsNone(ep['downloaded_at'])
        self.assertIsNone(ep['transcribed_at'])
        self.assertIsNone(ep['model_ready_at'])
        segments = pilot.read_rows(self.bundle / 'segments.jsonl')
        self.assertIn('Exampel', segments[0]['raw_text'])
        self.assertIn('Example', segments[0]['corrected_text'])
        for file in self.bundle.iterdir():
            self.assertNotIn('PRIVATE_TOKEN', file.read_text())
        candidate = pilot.read_rows(self.bundle / 'candidates.jsonl')[0]
        self.assertEqual(candidate['uncertainty_markers'], ['could'])
        self.assertEqual(len(candidate['segment_ids']), 2)

    def test_strict_cutoff_before_equal_after_and_expiry(self):
        dataset = self.reviewed()
        for index, (cutoff, expected) in enumerate([
            ('2024-03-03T23:59:59Z', 'not_available_before_cutoff'),
            ('2024-03-04T00:00:00Z', 'not_available_before_cutoff'),
            ('2024-03-04T00:00:01Z', None),
            ('2024-03-05T00:00:00Z', 'expired'),
        ]):
            rows = pilot.select(dataset, cutoff, self.root / f'cutoff{index}')
            self.assertEqual(rows[0]['exclusion_reason'], expected)
            self.assertEqual(rows[0]['eligible'], expected is None)

    def test_pending_is_never_model_ready(self):
        with patch.object(pilot, 'now', return_value='2024-03-04T00:00:00Z'):
            dataset = self.root / 'pending'
            pilot.finalize(self.bundle, self.bundle / 'review.json', dataset)
        rows = pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'selected')
        self.assertEqual(rows[0]['exclusion_reason'], 'not_accepted')
        self.assertIsNone(rows[0]['model_ready_at'])
        self.assertEqual((self.root / 'selected/eligible.jsonl').read_text(), '')

    def test_rejected_evidence_is_retained_in_audit(self):
        dataset = self.reviewed('rejected')
        rows = pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'audit')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['review']['status'], 'rejected')
        self.assertFalse(rows[0]['eligible'])

    def test_historical_assumptions_are_explicit_and_separate(self):
        dataset = self.reviewed()
        rows = pilot.select(dataset, '2024-03-02T00:00:00Z', self.root / 'historical', 2)
        self.assertTrue(rows[0]['eligible'])
        self.assertEqual(rows[0]['timing_basis'], 'historical_assumption')
        manifest = json.loads((self.root / 'historical/manifest.json').read_text())
        self.assertTrue(manifest['research_only'])
        self.assertEqual(manifest['historical_delay_hours'], 2)
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2024-03-02T00:00:00Z', self.root / 'invalid', float('nan'))

    def test_missing_mapping_blocks_acceptance(self):
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviewer'] = 'reviewer'
        review['reviews'][0].update(status='accepted', claim_text='Claim', notes='Checked')
        pilot.write_json(review_path, review)
        with self.assertRaises(ValueError):
            pilot.finalize(self.bundle, review_path, self.root / 'bad')

    def test_tampered_evidence_is_rejected(self):
        with (self.bundle / 'segments.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaises(ValueError):
            pilot.finalize(self.bundle, self.bundle / 'review.json', self.root / 'bad')

    def test_tampered_finalized_claims_are_rejected(self):
        dataset = self.reviewed()
        with (dataset / 'claims.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'bad')

    def test_historical_mention_is_not_a_current_signal(self):
        dataset = self.reviewed()
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviews'][0]['temporal_scope'] = 'historical'
        pilot.write_json(review_path, review)
        other = self.root / 'historical_claim'
        with patch.object(pilot, 'now', return_value='2024-03-04T00:00:00Z'):
            pilot.finalize(self.bundle, review_path, other)
        rows = pilot.select(other, '2024-03-04T01:00:00Z', self.root / 'historical_audit')
        self.assertEqual(rows[0]['exclusion_reason'], 'non_current_claim')

    def test_naive_cutoff_is_rejected(self):
        with self.assertRaises(ValueError):
            pilot.select(self.reviewed(), '2024-03-04T01:00:00', self.root / 'bad')

    def test_missing_transcript_remains_in_sampling_frame(self):
        with closing(sqlite3.connect(self.root / 'podcast.db')) as db, db:
            db.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)',
                       ('missing', 'demo', 'Daily Two', '2024-03-02T00:00:00Z', 'failed', None))
        out = self.root / 'missing'
        pilot.build(self.root, 'demo', 'Daily', 1, out)
        rows = pilot.read_rows(out / 'episodes.jsonl')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['audit_status'], 'missing_transcript')
        self.assertEqual(pilot.read_rows(out / 'candidates.jsonl'), [])

    def test_transcript_identity_mismatch_is_rejected(self):
        data = json.loads(self.source.read_text())
        data['guid'] = 'wrong'
        self.source.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            pilot.build(self.root, 'demo', 'Daily', 5, self.root / 'bad')

    def test_inconsistent_provenance_times_are_rejected(self):
        data = json.loads(self.source.read_text())
        data.update(downloaded_at='2024-03-02T10:00:00Z', transcribed_at='2024-03-02T09:00:00Z')
        self.source.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            pilot.build(self.root, 'demo', 'Daily', 5, self.root / 'bad')

    def test_existing_bundle_cannot_be_overwritten(self):
        with self.assertRaises(FileExistsError):
            pilot.build(self.root, 'demo', 'Daily', 5, self.bundle)


if __name__ == '__main__':
    unittest.main()
