"""Basic smoke tests for data utilities and solver interfaces."""

import pytest
from src.data_utils import classify_problem, split_examples, extract_query, label_by_answer


class TestClassifyProblem:
    """Tests for prompt-only keyword classifier (used on test set at inference)."""

    def test_bit_manipulation(self):
        assert classify_problem(
            "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers."
        ) == "bit_manipulation"

    def test_cipher(self):
        assert classify_problem(
            "In Alice's Wonderland, secret encryption rules are used on text."
        ) == "cipher"

    def test_unit_conversion(self):
        assert classify_problem(
            "In Alice's Wonderland, a secret unit conversion is applied to measurements."
        ) == "unit_conversion"

    def test_physics(self):
        assert classify_problem(
            "In Alice's Wonderland, the gravitational constant has been secretly changed."
        ) == "physics"

    def test_symbol_transform_fallback(self):
        assert classify_problem(
            "In Alice's Wonderland, numbers are secretly converted into a different numeral system."
        ) == "symbol_transform"


class TestLabelByAnswer:
    """Tests for answer-guided gold labeling (used to generate training labels)."""

    def test_binary_answer(self):
        assert label_by_answer("10010111", "bit manipulation prompt") == "bit_manipulation"

    def test_word_answer_is_cipher(self):
        assert label_by_answer("cat imagines book", "encryption prompt") == "cipher"

    def test_roman_numeral_is_symbol_transform(self):
        assert label_by_answer("XXXVIII", "numeral prompt") == "symbol_transform"

    def test_float_unit_conversion(self):
        assert label_by_answer("16.65", "unit conversion prompt") == "unit_conversion"

    def test_float_physics(self):
        assert label_by_answer("154.62", "gravitational constant prompt") == "physics"

    def test_float_numeric(self):
        assert label_by_answer("6644", "transformation rules prompt") == "numeric"


class TestSplitExamples:
    def test_basic(self):
        prompt = "01010001 -> 11011101\n00110011 -> 11001100\n11110000 ->"
        examples = split_examples(prompt)
        assert examples == [("01010001", "11011101"), ("00110011", "11001100")]

    def test_cipher(self):
        prompt = "hello -> ifmmp\nworld -> xpsme\ntest ->"
        examples = split_examples(prompt)
        assert len(examples) == 2
        assert examples[0] == ("hello", "ifmmp")


class TestExtractQuery:
    def test_bare_arrow(self):
        prompt = "01010001 -> 11011101\n00110011 ->"
        assert extract_query(prompt) == "00110011"

    def test_last_line(self):
        prompt = "hello -> world\nfoo"
        assert extract_query(prompt) == "foo"
