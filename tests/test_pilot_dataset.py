import importlib.util
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
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
                       ('https://secret.invalid/PRIVATE_TOKEN', 'demo', 'Daily One', '2024-03-01T00:00:00Z', 'transcribed',
                        str(self.source)))
        self.bundle = self.root / 'bundle'
        with patch.object(pilot, 'now', return_value='2024-03-03T00:00:00+00:00'):
            pilot.build(self.root, 'demo', 'Daily', 5, self.bundle)

    def reviewed(self, status='accepted'):
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviewer'] = 'test-reviewer'
        review['reviews'][0].update(status=status, claim_text='Alex Example might miss games.',
            certainty='speculation', temporal_scope='current',
            entities=[{'type': 'player', 'name': 'Alex Example', 'model_entity_id': 'test:123'}],
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
        rows = pilot.select(dataset, '2024-03-02T00:00:00Z', self.root / 'historical', 2, historical_validity_days=7)
        self.assertTrue(rows[0]['eligible'])
        self.assertEqual(rows[0]['timing_basis'], 'historical_assumption')
        manifest = json.loads((self.root / 'historical/manifest.json').read_text())
        self.assertTrue(manifest['research_only'])
        self.assertEqual(manifest['historical_delay_hours'], 2)
        self.assertEqual(manifest['historical_validity_days'], 7)
        for delay in (float('nan'), -1, 1e12):
            with self.assertRaises(ValueError):
                pilot.select(dataset, '2024-03-02T00:00:00Z', self.root / 'invalid', delay, historical_validity_days=7)

    # Invariant: in historical mode expiry comes only from the predeclared window, because
    # the reviewer's valid_until was written after the fact and can encode the outcome.
    def test_historical_expiry_ignores_reviewer_valid_until(self):
        dataset = self.reviewed()  # reviewer valid_until is 2024-03-05
        rows = pilot.select(dataset, '2024-03-06T00:00:00Z', self.root / 'policy', 2, historical_validity_days=7)
        self.assertTrue(rows[0]['eligible'])
        self.assertEqual(rows[0]['expiry_basis'], 'historical_policy')
        self.assertEqual(rows[0]['expires_at'], '2024-03-08T02:00:00+00:00')
        rows = pilot.select(dataset, '2024-03-08T02:00:00Z', self.root / 'policy_end', 2, historical_validity_days=7)
        self.assertEqual(rows[0]['exclusion_reason'], 'expired')

    def test_historical_selection_requires_validity_window_and_vice_versa(self):
        dataset = self.reviewed()
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2024-03-02T00:00:00Z', self.root / 'a', 2)
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'b', historical_validity_days=7)
        for days in (0, float('inf'), 1000):
            with self.assertRaises(ValueError):
                pilot.select(dataset, '2024-03-02T00:00:00Z', self.root / 'c', 2, historical_validity_days=days)

    def test_historical_unknown_publication_is_unavailable(self):
        with closing(sqlite3.connect(self.root / 'podcast.db')) as db, db:
            db.execute("UPDATE episodes SET published=NULL")
        self.bundle = self.root / 'unpublished'
        with patch.object(pilot, 'now', return_value='2024-03-03T00:00:00Z'):
            pilot.build(self.root, 'demo', 'Daily', 5, self.bundle)
        rows = pilot.select(self.reviewed(), '2024-03-04T01:00:00Z', self.root / 'h', 2, historical_validity_days=7)
        self.assertEqual(rows[0]['exclusion_reason'], 'unknown_availability')

    def test_audio_verification_gate_excludes_machine_crosschecks(self):
        dataset = self.reviewed()
        rows = pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'audio_gate', require_audio_checked=True)
        self.assertEqual(rows[0]['exclusion_reason'], 'audio_not_verified')
        manifest = json.loads((self.root / 'audio_gate/manifest.json').read_text())
        self.assertTrue(manifest['require_audio_checked'])
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviews'][0]['audio_checked'] = True
        pilot.write_json(review_path, review)
        checked = self.root / 'human_verified'
        with patch.object(pilot, 'now', return_value='2024-03-04T00:00:00Z'):
            pilot.finalize(self.bundle, review_path, checked)
        rows = pilot.select(checked, '2024-03-04T01:00:00Z', self.root / 'audio_pass', require_audio_checked=True)
        self.assertTrue(rows[0]['eligible'])

    def test_additional_evidence_is_resolved_from_frozen_segments(self):
        source = json.loads(self.source.read_text())
        source['segments'].append({'start': 15.0, 'end': 20.0, 'text': 'The named subject is Alex Example.'})
        self.source.write_text(json.dumps(source))
        self.bundle = self.root / 'expanded_bundle'
        with patch.object(pilot, 'now', return_value='2024-03-03T00:00:00Z'):
            pilot.build(self.root, 'demo', 'Daily', 5, self.bundle)
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        extra = pilot.read_rows(self.bundle / 'segments.jsonl')[-1]['segment_id']
        review['reviews'][0]['additional_segment_ids'] = [extra]
        pilot.write_json(review_path, review)
        rows = pilot.read_rows(self.reviewed() / 'claims.jsonl')
        self.assertEqual(rows[0]['reviewed_end_secs'], 20.0)
        self.assertIn('named subject', rows[0]['reviewed_raw_evidence'])
        self.assertEqual(len(rows[0]['segment_ids']), 2)
        self.assertEqual(len(rows[0]['reviewed_segment_ids']), 3)

    def test_unknown_additional_evidence_is_rejected(self):
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviews'][0]['additional_segment_ids'] = ['invented-segment']
        pilot.write_json(review_path, review)
        with self.assertRaises(ValueError):
            pilot.finalize(self.bundle, review_path, self.root / 'bad')

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
        self.reviewed()
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

    def test_failed_write_leaves_no_partial_output(self):
        out = self.root / 'partial'
        with patch.object(pilot, 'write_json', side_effect=OSError('disk full')), self.assertRaises(OSError):
            pilot.build(self.root, 'demo', 'Daily', 5, out)
        self.assertFalse(out.exists())
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith('.partial')], [])

    # Invariant: chrono's to_rfc3339 output (1-9 fraction digits) parses on every supported
    # Python and denotes the same instant, truncated to microseconds.
    def test_rust_rfc3339_timestamps_parse(self):
        for value, expected in [
            ('2026-09-27T01:02:03.123456789+00:00', '2026-09-27T01:02:03.123456+00:00'),
            ('2026-09-27T01:02:03.5+00:00', '2026-09-27T01:02:03.500000+00:00'),
            ('2026-09-27T11:02:03.123+10:00', '2026-09-27T01:02:03.123000+00:00'),
            ('2026-09-27T01:02:03Z', '2026-09-27T01:02:03+00:00'),
        ]:
            self.assertEqual(pilot.timestamp(value).isoformat(), expected)
        for bad in (12345, None, 'yesterday', '2026-09-27T01:02:03'):
            with self.assertRaises(ValueError):
                pilot.timestamp(bad)

    def test_build_accepts_nanosecond_provenance_times(self):
        data = json.loads(self.source.read_text())
        data.update(downloaded_at='2024-03-01T10:00:00.123456789+00:00',
                    transcribed_at='2024-03-01T10:05:00.5+00:00')
        self.source.write_text(json.dumps(data))
        out = self.root / 'nanos'
        with patch.object(pilot, 'now', return_value='2024-03-03T00:00:00+00:00'):
            pilot.build(self.root, 'demo', 'Daily', 5, out)
        self.assertEqual(pilot.read_rows(out / 'episodes.jsonl')[0]['transcribed_at'], '2024-03-01T10:05:00.5+00:00')

    def test_non_utc_cutoff_equal_to_availability_is_excluded(self):
        rows = pilot.select(self.reviewed(), '2024-03-04T10:00:00+10:00', self.root / 'tz')
        self.assertEqual(rows[0]['exclusion_reason'], 'not_available_before_cutoff')

    def test_malformed_review_fields_are_value_errors(self):
        review_path = self.bundle / 'review.json'
        original = json.loads(review_path.read_text())
        for index, change in enumerate([{'claim_text': None}, {'entities': ['Alex Example']},
                                        {'valid_until': 12345}, {'audio_checked': 'yes'},
                                        {'entities': [{'type': 'player', 'name': None, 'model_entity_id': 'x'}]}]):
            review = json.loads(json.dumps(original))
            review['reviewer'] = 'reviewer'
            review['reviews'][0].update(status='accepted', claim_text='Claim', notes='Checked', certainty='opinion',
                                        temporal_scope='current', valid_until='2024-03-05T00:00:00Z',
                                        entities=[{'type': 'player', 'name': 'A', 'model_entity_id': 'x'}])
            review['reviews'][0].update(change)
            pilot.write_json(review_path, review)
            with self.subTest(change=change), self.assertRaises(ValueError):
                pilot.finalize(self.bundle, review_path, self.root / f'bad{index}')

    def test_expiry_before_publication_is_rejected(self):
        review_path = self.bundle / 'review.json'
        review = json.loads(review_path.read_text())
        review['reviewer'] = 'reviewer'
        review['reviews'][0].update(status='accepted', claim_text='Claim', notes='Checked', certainty='opinion',
                                    temporal_scope='current', valid_until='2023-03-05T00:00:00Z',
                                    entities=[{'type': 'player', 'name': 'A', 'model_entity_id': 'x'}])
        pilot.write_json(review_path, review)
        with patch.object(pilot, 'now', return_value='2024-03-04T00:00:00Z'), self.assertRaises(ValueError):
            pilot.finalize(self.bundle, review_path, self.root / 'bad')

    # Invariant: a finalized dataset whose claims were edited is rejected even when the attacker
    # recomputes claims_sha256; a pinned manifest hash also rejects a fully consistent rewrite.
    def test_consistently_rewritten_claims_are_rejected(self):
        dataset = self.reviewed()
        rows = pilot.read_rows(dataset / 'claims.jsonl')
        rows[0].update(model_ready_at='2020-01-01T00:00:00+00:00')
        pilot.write_rows(dataset / 'claims.jsonl', rows)
        manifest = json.loads((dataset / 'manifest.json').read_text())
        manifest['claims_sha256'] = pilot.digest((dataset / 'claims.jsonl').read_bytes())
        pilot.write_json(dataset / 'manifest.json', manifest)
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2021-01-01T00:00:00Z', self.root / 'bad')

    def test_review_copy_edits_are_rejected(self):
        dataset = self.reviewed()
        review = json.loads((dataset / 'review.json').read_text())
        review['reviews'][0]['valid_until'] = '2030-01-01T00:00:00Z'
        pilot.write_json(dataset / 'review.json', review)
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'bad')

    def test_pinned_manifest_hash(self):
        dataset = self.reviewed()
        pin = pilot.digest((dataset / 'manifest.json').read_bytes())
        rows = pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'pinned', expected_manifest_sha256=pin)
        self.assertTrue(rows[0]['eligible'])
        manifest = json.loads((self.root / 'pinned/manifest.json').read_text())
        self.assertEqual(manifest['source_manifest_sha256'], pin)
        with self.assertRaises(ValueError):
            pilot.select(dataset, '2024-03-04T01:00:00Z', self.root / 'wrong', expected_manifest_sha256='0' * 64)

    def test_manifest_records_extractor_and_vocab_hashes(self):
        manifest = json.loads((self.bundle / 'manifest.json').read_text())
        self.assertIsNone(manifest['vocab_sha256'])
        self.assertEqual(len(manifest['extractor_sha256']), 64)
        (self.root / 'vocab').mkdir()
        (self.root / 'vocab/demo.txt').write_text('Alex Example\n')
        out = self.root / 'with_vocab'
        pilot.build(self.root, 'demo', 'Daily', 5, out)
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(manifest['vocab_sha256'], pilot.digest(b'Alex Example\n'))
        self.assertEqual(pilot.read_rows(out / 'candidates.jsonl')[0]['entity_suggestions'], ['Alex Example'])

    def test_null_title_is_not_a_crash(self):
        with closing(sqlite3.connect(self.root / 'podcast.db')) as db, db:
            db.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)', ('untitled', 'demo', None, '2024-03-02T00:00:00Z', 'new', None))
        pilot.build(self.root, 'demo', 'Daily', 5, self.root / 'untitled')


if __name__ == '__main__':
    unittest.main()
