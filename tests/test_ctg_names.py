import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('ctg_names', Path(__file__).parents[1] / 'scripts/ctg_names.py')
ctg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctg)


class NameExtractionTests(unittest.TestCase):
    def extract(self, data):
        names = set()
        ctg.walk(data, names)
        return names

    def test_accented_and_punctuated_names_are_kept(self):
        data = [{'player': 'Nikola Jokić'}, {'name': 'Luka Dončić'}, {'player_name': "De'Aaron Fox"},
                {'Name': 'Jaren Jackson Jr.'}, {'player': 'Shai Gilgeous-Alexander'}]
        self.assertEqual(self.extract(data), {'Nikola Jokić', 'Luka Dončić', "De'Aaron Fox",
                                              'Jaren Jackson Jr.', 'Shai Gilgeous-Alexander'})

    def test_non_player_name_keys_are_ignored(self):
        data = {'team_name': 'Los Angeles Lakers', 'arena_name': 'Crypto Arena', 'stats': [{'player': 'Alex Example'}]}
        self.assertEqual(self.extract(data), {'Alex Example'})

    def test_non_names_are_rejected(self):
        data = [{'name': 'lowercase name'}, {'name': 'Single'}, {'name': 'Player 23'}, {'name': ' '}]
        self.assertEqual(self.extract(data), set())


if __name__ == '__main__':
    unittest.main()
