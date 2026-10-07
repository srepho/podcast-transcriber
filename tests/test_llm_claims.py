import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))  # pilot_dataset imports llm_claims by module name, as when run as a script


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load('pilot_dataset')
llm = load('llm_claims')
reviewer = load('review_claims')


def claim(start=0, end=1, **overrides):
    return {'claim_text': 'Alex Example will miss two weeks with an ankle sprain.', 'claim_type': 'injury',
            'certainty': 'reported_fact', 'temporal_scope': 'current',
            'entities': [{'type': 'player', 'name': 'Alex Example'}, {'type': 'team', 'name': 'Boston Celtics'}],
            'segment_start': start, 'segment_end': end, 'asr_uncertain': False, **overrides}


class ProposalTests(unittest.TestCase):
    def test_impossible_evidence_and_bad_enums_are_dropped_not_repaired(self):
        # Invariant: every kept proposal cites segments that exist, within MAX_SPAN, with valid labels.
        response = {'claims': [claim(), claim(start=-1), claim(start=3, end=2), claim(end=50),
                               claim(start=0, end=llm.MAX_SPAN), claim(claim_type='vibes'), claim(claim_text='  ')]}
        kept = llm.proposals(response, n_segments=40)
        self.assertEqual(len(kept), 1)
        self.assertEqual((kept[0]['segment_start'], kept[0]['segment_end']), (0, 1))

    def test_entities_without_a_name_or_known_type_are_dropped(self):
        kept = llm.proposals({'claims': [claim(entities=[{'type': 'coach', 'name': 'X'}, {'type': 'player', 'name': ' '},
                                                         {'type': 'team', 'name': 'Utah Jazz'}])]}, 2)
        self.assertEqual(kept[0]['entities'], [{'type': 'team', 'name': 'Utah Jazz'}])


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = llm.Catalogue({
            llm.name_key('Nikola Jokić'): [('3112335', True)],
            llm.name_key('Kevin Porter'): [('1', False), ('2', True)],   # one active namesake wins
            llm.name_key('Chris Johnson'): [('5', False), ('6', False)],  # still ambiguous
        })

    def test_exact_match_ignores_case_accents_and_punctuation(self):
        self.assertEqual(self.catalogue.resolve('player', 'nikola jokic'), 'espn:athlete:3112335')

    def test_namesakes_resolve_only_when_unambiguous(self):
        self.assertEqual(self.catalogue.resolve('player', 'Kevin Porter'), 'espn:athlete:2')
        self.assertIsNone(self.catalogue.resolve('player', 'Chris Johnson'))

    def test_no_fuzzy_matching(self):
        self.assertIsNone(self.catalogue.resolve('player', 'Nikola Yokic'))

    def test_teams_by_full_name_nickname_and_alias(self):
        for name in ('Boston Celtics', 'Celtics', 'Sixers', 'Los Angeles Clippers', 'LA Clippers'):
            self.assertTrue(self.catalogue.resolve('team', name).startswith('espn:team:'), name)
        self.assertEqual(self.catalogue.resolve('team', 'Sixers'), 'espn:team:20')
        self.assertIsNone(self.catalogue.resolve('team', 'Seattle SuperSonics'))

    def test_load_reads_athletes_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'athletes.jsonl'
            path.write_text('{"athlete_id": "1966", "name": "LeBron James", "active": true}\n')
            self.assertEqual(llm.Catalogue.load(path).resolve('player', 'LeBron James'), 'espn:athlete:1966')
            self.assertIsNone(llm.Catalogue.load(Path(d) / 'missing.jsonl').resolve('player', 'LeBron James'))


class ClaudeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with closing(sqlite3.connect(self.root / 'podcast.db')) as db, db:
            db.execute('CREATE TABLE episodes (guid TEXT, feed_name TEXT, title TEXT, published TEXT, status TEXT, transcript_path TEXT)')
            for n, published in ((1, '2024-03-01T00:00:00Z'), (2, '2024-03-02T00:00:00Z'), (3, '2024-01-01T00:00:00Z')):
                source = self.root / f'ep{n}.json'
                source.write_text(json.dumps({'guid': f'g{n}', 'feed': 'demo', 'model': 'base.en', 'segments': [
                    {'start': 0.0, 'end': 5.0, 'text': 'Alex Example sprained his ankle.'},
                    {'start': 5.0, 'end': 9.0, 'text': 'He will miss two weeks, the team said.'}]}))
                db.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?)',
                           (f'g{n}', 'demo', f'Show {n}', published, 'transcribed', str(source)))
        self.catalogue = llm.Catalogue({llm.name_key('Alex Example'): [('123', True)]})
        self.prompts = []

    def call(self, prompt):
        self.prompts.append(prompt)
        return {'claims': [claim()]}

    def build(self, out, call=None, stamp='2024-03-03T00:00:00+00:00'):
        with patch.object(pilot, 'now', return_value=stamp):
            return pilot.build(self.root, 'demo', '', 10, self.root / out, extractor='llm', unprocessed=True,
                               published_after='2024-02-01T00:00:00Z', catalogue=self.catalogue, call=call or self.call)

    def test_proposals_become_pending_prefilled_reviews(self):
        manifest = self.build('auto1')
        self.assertEqual(manifest['extractor_version'], llm.EXTRACTOR_VERSION)
        self.assertEqual(manifest['counts']['episodes.jsonl'], 2)  # the January episode is before the floor
        self.assertTrue(all('[0] Alex Example sprained his ankle.' in p for p in self.prompts))
        review = json.loads((self.root / 'auto1/review.json').read_text())
        item = review['reviews'][0]
        self.assertEqual(item['status'], 'pending')
        self.assertEqual(item['entities'][0]['model_entity_id'], 'espn:athlete:123')
        self.assertEqual(item['entities'][1]['model_entity_id'], 'espn:team:2')
        # Expiry policy: publication + EXPIRY_DAYS[type], fixed before evaluation.
        self.assertEqual({r['valid_until'] for r in review['reviews']},
                         {'2024-03-08T00:00:00+00:00', '2024-03-09T00:00:00+00:00'})
        candidate = pilot.read_rows(self.root / 'auto1/candidates.jsonl')[0]
        self.assertEqual(len(candidate['segment_ids']), 2)
        self.assertEqual(candidate['proposal']['certainty'], 'reported_fact')

    def test_unprocessed_never_extracts_an_episode_twice(self):
        self.build('auto1')
        self.assertIsNone(self.build('auto2'))
        self.assertFalse((self.root / 'auto2').exists())
        self.assertEqual(len(self.prompts), 2)

    def test_failed_episode_is_kept_visible_and_retried(self):
        def flaky(prompt):
            if 'Show 2' in prompt:
                raise RuntimeError('extraction refused')
            return self.call(prompt)
        manifest = self.build('auto1', call=flaky)
        self.assertEqual(len(manifest['extraction_failures']), 1)
        statuses = sorted(e['audit_status'] for e in pilot.read_rows(self.root / 'auto1/episodes.jsonl'))
        self.assertEqual(statuses, ['extraction_failed', 'ready_for_review'])
        retry = self.build('auto2', stamp='2024-03-03T06:00:00+00:00')
        self.assertEqual(retry['counts']['episodes.jsonl'], 1)
        self.assertEqual(pilot.read_rows(self.root / 'auto2/episodes.jsonl')[0]['title'], 'Show 2')

    def test_unprocessed_requires_a_floor(self):
        with self.assertRaises(ValueError):
            pilot.build(self.root, 'demo', '', 5, self.root / 'x', unprocessed=True)

    def test_review_finalize_select_round_trip(self):
        self.build('auto1')
        bundle = self.root / 'auto1'
        answers = iter(['a', 'r'])
        with patch.object(reviewer, 'save', wraps=reviewer.save):
            reviewer.review(bundle, 'tester', read=lambda _: next(answers))
        dataset = self.root / 'auto1-reviewed'
        with patch.object(pilot, 'now', return_value='2024-03-04T00:00:00+00:00'):
            pilot.finalize(bundle, bundle / 'review.json', dataset)
        rows = pilot.select(dataset, '2024-03-05T00:00:00Z', self.root / 'cut')
        self.assertEqual(sorted(r['eligible'] for r in rows), [False, True])
        accepted = next(r for r in rows if r['eligible'])
        self.assertEqual(accepted['review']['notes'], reviewer.ACCEPT_NOTE)
        self.assertFalse(accepted['review']['audio_checked'])


class FakeOpenAI:
    """Stands in for openai.OpenAI; records the request and returns a canned reply."""
    reply, finish, requests = '{"claims": []}', 'stop', []

    def __init__(self, api_key, base_url):
        FakeOpenAI.requests.append({'api_key': api_key, 'base_url': base_url})
        self.chat = self
        self.completions = self

    def create(self, **request):
        FakeOpenAI.requests[-1].update(request)
        message = type('M', (), {'content': FakeOpenAI.reply, 'refusal': None})
        return type('R', (), {'choices': [type('C', (), {'message': message, 'finish_reason': FakeOpenAI.finish})]})


class ProviderTests(unittest.TestCase):
    def setUp(self):
        FakeOpenAI.reply, FakeOpenAI.finish, FakeOpenAI.requests = '{"claims": []}', 'stop', []
        patcher = patch.dict('sys.modules', {'openai': type('openai', (), {'OpenAI': FakeOpenAI})})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_non_anthropic_providers_need_an_explicit_model(self):
        self.assertEqual(llm.Extractor().name, 'anthropic:claude-opus-5-5')
        for provider in ('openai', 'deepseek', 'qwen', 'moonshot', 'zhipu'):
            with self.assertRaises(ValueError):
                llm.Extractor(provider)
        with self.assertRaises(ValueError):
            llm.Extractor('compatible', 'some-model')  # no base URL
        with self.assertRaises(ValueError):
            llm.Extractor('nonsense', 'x')

    def test_fingerprint_distinguishes_provider_model_and_endpoint(self):
        self.assertEqual(llm.Extractor('deepseek', 'm').fingerprint()['options'], {'reasoning_effort': 'low'})
        prints = {json.dumps(llm.Extractor(*args).fingerprint(), sort_keys=True) for args in
                  [('deepseek', 'm1'), ('deepseek', 'm2'), ('qwen', 'm1'), ('deepseek', 'm1', 'https://other.example/v1')]}
        self.assertEqual(len(prints), 4)

    def test_openai_uses_strict_schema(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'k'}):
            llm.Extractor('openai', 'gpt-test')('prompt')
        request = FakeOpenAI.requests[-1]
        self.assertEqual(request['response_format']['type'], 'json_schema')
        self.assertTrue(request['response_format']['json_schema']['strict'])
        self.assertIn('max_completion_tokens', request)
        self.assertNotIn('JSON Schema', request['messages'][0]['content'])

    def test_compatible_providers_get_json_mode_and_the_schema_in_the_prompt(self):
        FakeOpenAI.reply = '```json\n{"claims": [{"claim_text": "x"}]}\n```'
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'k'}):
            data = llm.Extractor('deepseek', 'deepseek-test')('prompt')
        request = FakeOpenAI.requests[-1]
        self.assertEqual(request['base_url'], 'https://api.deepseek.com')
        self.assertEqual(request['response_format'], {'type': 'json_object'})
        self.assertIn('JSON Schema', request['messages'][0]['content'])
        self.assertEqual(data, {'claims': [{'claim_text': 'x'}]})
        # DeepSeek reasoning is bounded: low effort and a larger output budget (see PROVIDERS).
        self.assertEqual(request['reasoning_effort'], 'low')
        self.assertEqual(request['max_tokens'], 32000)
        self.assertEqual(llm.proposals(data, 5), [])  # unvalidated JSON is still filtered

    def test_missing_key_truncation_and_non_object_replies_fail_loudly(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(RuntimeError):
            llm.Extractor('qwen', 'q')('prompt')
        with patch.dict('os.environ', {'MOONSHOT_API_KEY': 'k'}):
            FakeOpenAI.finish = 'length'
            with self.assertRaises(RuntimeError):
                llm.Extractor('moonshot', 'k')('prompt')
            FakeOpenAI.finish, FakeOpenAI.reply = 'stop', '[1, 2]'
            with self.assertRaises(ValueError):
                llm.Extractor('moonshot', 'k')('prompt')

    def test_provider_and_model_are_recorded_on_every_candidate(self):
        segments = [{'index': 0, 'segment_id': 'e:0', 'start_secs': 0.0, 'end_secs': 1.0, 'raw_text': 'a', 'corrected_text': 'a'},
                    {'index': 1, 'segment_id': 'e:1', 'start_secs': 1.0, 'end_secs': 2.0, 'raw_text': 'b', 'corrected_text': 'b'}]
        args = ('e', 'sha', 'Show', '2024-03-01T00:00:00Z', segments, llm.Catalogue({}), lambda v: json.dumps(v), pilot.timestamp,
                lambda _: {'claims': [claim()]})
        one, _ = llm.episode_candidates(*args, 'deepseek:m1')
        two, _ = llm.episode_candidates(*args, 'qwen:m1')
        self.assertEqual(one[0]['extractor_model'], 'deepseek:m1')
        self.assertNotEqual(one[0]['candidate_id'], two[0]['candidate_id'])


class ReviewCommandTests(unittest.TestCase):
    def item(self):
        return {'status': 'pending', 'claim_text': 'X', 'certainty': 'opinion', 'valid_until': None, 'notes': '',
                'entities': [{'type': 'player', 'name': 'Alex Example', 'model_entity_id': ''}]}

    def test_accept_is_blocked_until_ids_and_expiry_are_set(self):
        item = self.item()
        done, message = reviewer.apply(item, 'a', '2024-03-01T00:00:00Z', 'me')
        self.assertFalse(done)
        self.assertIn('model id for Alex Example', message)
        self.assertIn('expiry', message)
        self.assertFalse(reviewer.apply(item, 'i 0 espn:athlete:9', None, 'me')[0])
        self.assertFalse(reviewer.apply(item, 'x 3', '2024-03-01T00:00:00Z', 'me')[0])
        self.assertEqual(item['valid_until'], '2024-03-04T00:00:00+00:00')
        self.assertEqual(reviewer.apply(item, 'a', '2024-03-01T00:00:00Z', 'me'), (True, 'accepted'))

    def test_bad_edits_are_refused(self):
        item = self.item()
        for command in ('i 0 123', 'i 5 espn:athlete:1', 'x -2', 'x soon', 'c certain', 'zz'):
            self.assertFalse(reviewer.apply(item, command, '2024-03-01T00:00:00Z', 'me')[0], command)
        self.assertEqual(item['entities'][0]['model_entity_id'], '')
        self.assertIsNone(item['valid_until'])
        self.assertEqual(item['certainty'], 'opinion')


if __name__ == '__main__':
    unittest.main()


daily = load('daily')


class DailyRunnerTests(unittest.TestCase):
    def test_dotenv_reads_only_the_requested_variable(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '.env'
            path.write_text('OTHER_SECRET=nope\nexport DEEPSEEK_API_KEY="first"\n# DEEPSEEK_API_KEY=commented\nDEEPSEEK_API_KEY=\'last\'\n')
            self.assertEqual(daily.dotenv_value(path, 'DEEPSEEK_API_KEY'), 'last')
            self.assertIsNone(daily.dotenv_value(path, 'MISSING'))
            with patch.dict('os.environ', {}, clear=True), \
                    patch.object(daily.subprocess, 'run', return_value=type('P', (), {'returncode': 44, 'stdout': ''})):
                self.assertTrue(daily.load_key('deepseek', path))
                self.assertEqual(daily.os.environ['DEEPSEEK_API_KEY'], 'last')
                self.assertNotIn('OTHER_SECRET', daily.os.environ)

    def test_no_key_anywhere_is_reported(self):
        with patch.dict('os.environ', {}, clear=True), \
                patch.object(daily.subprocess, 'run', return_value=type('P', (), {'returncode': 44, 'stdout': ''})):
            self.assertFalse(daily.load_key('qwen', None))

    def test_summary(self):
        manifest = {'counts': {'candidates.jsonl': 7, 'episodes.jsonl': 2}, 'extraction_failures': [{}]}
        self.assertEqual(daily.summary(manifest), '7 claims from 2 episodes, 1 failed')


class CatalogueMatchingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'history.jsonl').write_text('\n'.join(json.dumps(r) for r in [
            {'athlete_id': '1', 'name': 'Gary Trent', 'active': False},
            {'athlete_id': '2', 'name': 'Gary Trent Jr.', 'active': True},
            {'athlete_id': '3', 'name': 'Jimmy Butler III', 'active': True},
            {'athlete_id': '4', 'name': 'Glenn Robinson', 'active': False},
            {'athlete_id': '5', 'name': 'Glenn Robinson III', 'active': False},
            {'athlete_id': '6', 'name': 'Herbert Jones', 'active': True}]) + '\n')
        (root / 'rosters.jsonl').write_text(json.dumps({'athlete_id': '7', 'name': 'Thomas Sorber', 'active': True}) + '\n'
                                            + json.dumps({'athlete_id': '6', 'name': 'Herb Jones', 'active': True}) + '\n')
        (root / 'aliases.txt').write_text('# nicknames\nLu Dort => Luguentz Dort\nHerb Jones => Herbert Jones\n')
        self.catalogue = llm.Catalogue.load([root / 'history.jsonl', root / 'rosters.jsonl', root / 'missing.jsonl'],
                                            root / 'aliases.txt')

    def test_suffix_is_optional_when_unambiguous(self):
        # Invariant: a name spoken without its suffix reaches the single active suffix variant.
        self.assertEqual(self.catalogue.resolve('player', 'Jimmy Butler'), 'espn:athlete:3')
        self.assertEqual(self.catalogue.resolve('player', 'Gary Trent'), 'espn:athlete:2')  # not the retired father
        self.assertEqual(self.catalogue.resolve('player', 'Gary Trent Jr.'), 'espn:athlete:2')
        self.assertIsNone(self.catalogue.resolve('player', 'Glenn Robinson'))  # two inactive candidates

    def test_a_spoken_suffix_must_match_exactly(self):
        self.assertIsNone(self.catalogue.resolve('player', 'Herbert Jones Jr.'))

    def test_later_files_add_players_and_aliases_map_nicknames(self):
        self.assertEqual(self.catalogue.resolve('player', 'Thomas Sorber'), 'espn:athlete:7')
        self.assertEqual(self.catalogue.resolve('player', 'Herb Jones'), 'espn:athlete:6')
        self.assertIsNone(self.catalogue.resolve('player', 'Lu Dort'))  # alias target not in the catalogue


class DuplicateReviewTests(unittest.TestCase):
    def entry(self, cid, text, status='pending', kind='contract', ids=('espn:athlete:9', 'espn:team:8')):
        return {'candidate_id': cid, 'status': status, 'claim_type': kind, 'claim_text': text,
                'entities': [{'type': 'player' if i.startswith('espn:athlete') else 'team', 'name': i, 'model_entity_id': i}
                             for i in ids]}

    def test_related_needs_same_type_and_players_and_ranks_by_text(self):
        item = self.entry('new', 'Jalen Duren signed a five-year, $200 million extension.')
        decided = [self.entry('a', 'Jalen Duren signed a five-year $200 million extension with Detroit.', 'accepted'),
                   self.entry('b', 'The Pistons worried about Duren conditioning.', 'rejected'),
                   self.entry('c', 'Jalen Duren signed.', 'accepted', kind='transaction'),
                   self.entry('d', 'Same player, other team.', 'accepted', ids=('espn:athlete:9',)),
                   self.entry('e', 'Other player.', 'accepted', ids=('espn:athlete:1', 'espn:team:8'))]
        self.assertEqual([d['candidate_id'] for d in reviewer.related(item, decided)], ['a', 'd'])
        self.assertEqual(reviewer.related(self.entry('x', 'Unmapped', ids=()), decided), [])

    def test_duplicate_command(self):
        item, original = self.entry('new', 'x'), self.entry('a', 'Duren signed.', 'accepted')
        self.assertEqual(reviewer.apply(item, 'd', None, 'me'), (False, 'no related decided claim to mark this a duplicate of'))
        self.assertEqual(reviewer.apply(item, 'd', None, 'me', [original]), (True, 'rejected as duplicate'))
        self.assertEqual(item['status'], 'rejected')
        self.assertIn('Duplicate of a', item['notes'])

    def test_remap_fills_only_pending_unmapped_entities(self):
        catalogue = llm.Catalogue({llm.name_key('Alex Example'): [('1', True)]})
        data = {'reviews': [
            {'status': 'pending', 'entities': [{'type': 'player', 'name': 'Alex Example', 'model_entity_id': ''}]},
            {'status': 'rejected', 'entities': [{'type': 'player', 'name': 'Alex Example', 'model_entity_id': ''}]}]}
        self.assertEqual(reviewer.remap(data, catalogue), 1)
        self.assertEqual(data['reviews'][0]['entities'][0]['model_entity_id'], 'espn:athlete:1')
        self.assertEqual(data['reviews'][1]['entities'][0]['model_entity_id'], '')

    def test_other_bundles_decisions_are_seen_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ('one', 'one-reviewed', 'mine'):
                (root / name).mkdir()
            decided = {'reviews': [self.entry('a', 'Duren signed.', 'accepted'), self.entry('p', 'pending one')]}
            for name in ('one', 'one-reviewed'):
                (root / name / 'review.json').write_text(json.dumps(decided))
            (root / 'mine' / 'review.json').write_text(json.dumps({'reviews': [self.entry('m', 'mine', 'accepted')]}))
            self.assertEqual([r['candidate_id'] for r in reviewer.decided_elsewhere(root / 'mine')], ['a'])
