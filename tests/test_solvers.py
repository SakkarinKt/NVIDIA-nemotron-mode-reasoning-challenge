"""Basic smoke tests for data utilities and solver interfaces."""

import pytest
from src.data_utils import classify_problem, split_examples, extract_query


class TestClassifyProblem:
    def test_bit_manipulation(self):
        assert classify_problem("01010001 xor 11001100 -> result") == "bit_manipulation"

    def test_cipher(self):
        assert classify_problem("encrypt the word hello") == "cipher"

    def test_unit_conversion(self):
        assert classify_problem("convert 10 meter to feet") == "unit_conversion"

    def test_number_conversion(self):
        assert classify_problem("11 -> XI using roman numeral") == "number_conversion"

    def test_symbol_transform_fallback(self):
        assert classify_problem("@ # $ -> something") == "symbol_transform"


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
