import unittest
from collections import Counter

from llm_router.training.dataset import records, english_records
from llm_router.features import CODE_CREATION


class DatasetTests(unittest.TestCase):
    def test_counts_uniqueness_and_difficulty(self):
        self.assertEqual(Counter(row["label"] for row in records),
                         {"E2B": 500, "E4B": 500, "12B": 500})
        self.assertEqual(len({row["text"].casefold() for row in records}), 1500)
        bands = {"E2B": range(0, 3), "E4B": range(3, 6), "12B": range(6, 10)}
        self.assertTrue(all(row["difficulty"] in bands[row["label"]] for row in records))
        family_counts = Counter(row["family"] for row in records)
        self.assertEqual(len(family_counts), 300)
        self.assertTrue(all(count == 5 for count in family_counts.values()))

    def test_e2b_has_no_code_creation_and_calculator_is_e4b(self):
        self.assertFalse(any(CODE_CREATION.search(row["text"])
                             for row in records if row["label"] == "E2B"))
        self.assertTrue(any("калькулятор" in row["text"].lower()
                            for row in records if row["label"] == "E4B"))
        self.assertFalse(any("калькулятор" in row["text"].lower()
                             for row in records if row["label"] == "E2B"))
        self.assertIsNotNone(CODE_CREATION.search("Нужен простой калькулятор на питоне."))
        self.assertIsNone(CODE_CREATION.search("Что такое функция в Python?"))

    def test_english_counts_uniqueness_and_family_groups(self):
        self.assertEqual(Counter(row["label"] for row in english_records),
                         {"E2B": 500, "E4B": 500, "12B": 500})
        self.assertEqual(len({row["text"].casefold() for row in english_records}), 1500)
        family_counts = Counter(row["family"] for row in english_records)
        self.assertEqual(len(family_counts), 300)
        self.assertTrue(all(count == 5 for count in family_counts.values()))


if __name__ == "__main__":
    unittest.main()
