"""Unit tests for typed Question factory methods and schemas."""

import pytest
from anydecision.core.question import Question
from anydecision.core.types import AnswerType, OptionDefinition, ReadoutStrategy


def test_question_binary():
    q = Question.binary("Is this transaction fraudulent?")
    assert q.answer_type == AnswerType.BINARY
    assert q.option_keys() == ["yes", "no"]
    assert q.validate_option("yes") is True
    assert q.validate_option("maybe") is False
    assert q.readout_strategy == ReadoutStrategy.NEXT_TOKEN


def test_question_choice():
    q = Question.choice("Choose tag", ["billing", "technical", "sales", "other"])
    assert q.answer_type == AnswerType.CHOICE
    assert len(q.options) == 4
    assert q.validate_option("billing") is True
    assert q.validate_option("fraud") is False


def test_question_multi_token_detection():
    q = Question.choice("Choose action", ["quick reply", "escalate to tier 2"])
    assert q.readout_strategy == ReadoutStrategy.MULTI_TOKEN_SEQUENCE


def test_question_ordinal():
    levels = ["low", "medium", "high", "critical"]
    q = Question.ordinal("Severity level?", levels)
    assert q.answer_type == AnswerType.ORDINAL
    assert len(q.options) == 4
    assert q.options[0].ordinal_rank == 0
    assert q.options[3].ordinal_rank == 3


def test_question_score():
    q = Question.score("Rate from 0 to 5", minimum=0, maximum=5)
    assert q.answer_type == AnswerType.NUMERIC_SCORE
    assert len(q.options) == 6
    assert q.options[0].numeric_value == 0.0
    assert q.options[5].numeric_value == 5.0


def test_question_deterministic_id():
    id1 = Question.generate_id("Is this valid?", ["yes", "no"])
    id2 = Question.generate_id("Is this valid?", ["no", "yes"])
    assert id1 == id2

