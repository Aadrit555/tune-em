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


def test_question_validation_empty_text():
    with pytest.raises(ValueError, match="cannot be empty or blank"):
        Question.choice("", ["opt1", "opt2"])

    with pytest.raises(ValueError, match="cannot be empty or blank"):
        Question.choice("   ", ["opt1", "opt2"])


def test_question_validation_insufficient_options():
    with pytest.raises(ValueError, match="requires at least 2 candidate options"):
        Question(text="Is this valid?", options=[OptionDefinition(key="yes", label="yes")])

    # allow_custom_readout=True should bypass option count check
    custom_q = Question.custom(
        text="Special prompt",
        options=[OptionDefinition(key="single", label="Single Option")],
    )
    assert len(custom_q.options) == 1
    assert custom_q.allow_custom_readout is True


def test_question_validation_duplicate_keys():
    with pytest.raises(ValueError, match="Duplicate option keys detected"):
        Question(
            text="Choose option",
            options=[
                OptionDefinition(key="dup", label="First"),
                OptionDefinition(key="dup", label="Second"),
            ],
        )


def test_question_validation_empty_labels():
    with pytest.raises(ValueError, match="empty or blank label"):
        Question(
            text="Choose option",
            options=[
                OptionDefinition(key="opt1", label="Valid"),
                OptionDefinition(key="opt2", label="   "),
            ],
        )


def test_question_validation_ordinal_ranks():
    with pytest.raises(ValueError, match="must specify an integer ordinal_rank"):
        Question(
            text="Rank?",
            answer_type=AnswerType.ORDINAL,
            options=[
                OptionDefinition(key="low", label="Low", ordinal_rank=0),
                OptionDefinition(key="high", label="High", ordinal_rank=None),
            ],
        )

    with pytest.raises(ValueError, match="Duplicate ordinal ranks detected"):
        Question(
            text="Rank?",
            answer_type=AnswerType.ORDINAL,
            options=[
                OptionDefinition(key="low", label="Low", ordinal_rank=0),
                OptionDefinition(key="high", label="High", ordinal_rank=0),
            ],
        )


def test_question_validation_numeric_score():
    with pytest.raises(ValueError, match="must be strictly less than maximum"):
        Question.score("Score?", minimum=10, maximum=5)

    with pytest.raises(ValueError, match="must be strictly positive"):
        Question.score("Score?", minimum=0, maximum=10, step=0)


def test_question_multi_choice_independent_evaluation():
    from anydecision.core.engine import DecisionEngine

    engine = DecisionEngine(model="mock")
    q = Question.multi_choice(
        text="Which programming languages are used in this project?",
        choices=["Python", "Rust", "Go"],
        threshold=0.40,
    )
    assert q.answer_type == AnswerType.MULTI_CHOICE
    assert len(q.options) == 3

    decision = engine.decide(q)
    assert decision.method == "independent_multilabel"
    assert decision.labels is not None
    assert isinstance(decision.labels, list)
    assert set(decision.probabilities.keys()) == {"Python", "Rust", "Go"}
    # Probabilities are independent Bernoulli evaluations, each between 0.0 and 1.0
    for p in decision.probabilities.values():
        assert 0.0 <= p <= 1.0
    # Selected labels must match threshold
    for lbl in decision.labels:
        assert decision.probabilities[lbl] >= 0.40
    assert decision.backend_calls == 3

