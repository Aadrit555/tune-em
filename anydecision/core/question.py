"""Typed question representation and factory methods."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

from anydecision.core.types import AnswerType, OptionDefinition, ReadoutStrategy


class Question(BaseModel):
    """A strongly typed question with discrete candidate options and validation rules.

    Does NOT depend on generation. Instead, candidates represent targets whose direct
    probabilities will be evaluated and normalized by the decision engine.
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable identifier for caching, tracing, and dataset tracking."
    )
    text: str = Field(description="The primary question or prompt body.")
    answer_type: AnswerType = Field(default=AnswerType.CHOICE, description="Semantic answer type.")
    options: List[OptionDefinition] = Field(
        default_factory=list,
        description="Candidate answer options evaluated directly in the model output space."
    )
    readout_strategy: ReadoutStrategy = Field(
        default=ReadoutStrategy.NEXT_TOKEN,
        description="Whether next-token or multi-token sequence scoring is applied."
    )
    allow_custom_readout: bool = Field(
        default=False,
        description="Flag indicating if a custom readout hook should be invoked."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined context, tags, or operational metadata."
    )
    context: Optional[str] = Field(
        default=None,
        description="Optional preceding context or document text."
    )

    def option_keys(self) -> List[str]:
        """Return list of option keys."""
        return [opt.key for opt in self.options]

    def option_labels(self) -> List[str]:
        """Return list of option labels/text."""
        return [opt.label for opt in self.options]

    def get_option_by_key(self, key: str) -> Optional[OptionDefinition]:
        """Lookup an option definition by key."""
        for opt in self.options:
            if opt.key == key:
                return opt
        return None

    def validate_option(self, key: str) -> bool:
        """Validate if a candidate key belongs to this question's defined options."""
        return key in self.option_keys()

    @classmethod
    def generate_id(cls, text: str, options: Sequence[str]) -> str:
        """Compute a deterministic hash ID from text and candidate options."""
        content = f"{text.strip()}::" + "::".join(sorted(opt.strip() for opt in options))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def binary(
        cls,
        text: str,
        yes_option: str = "yes",
        no_option: str = "no",
        question_id: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Question:
        """Construct a binary (yes/no or true/false) question."""
        qid = question_id or cls.generate_id(text, [yes_option, no_option])
        options = [
            OptionDefinition(key=yes_option, label=yes_option),
            OptionDefinition(key=no_option, label=no_option),
        ]
        # Multi-token detection
        readout = (
            ReadoutStrategy.MULTI_TOKEN_SEQUENCE
            if (" " in yes_option or " " in no_option)
            else ReadoutStrategy.NEXT_TOKEN
        )
        return cls(
            id=qid,
            text=text,
            answer_type=AnswerType.BINARY,
            options=options,
            readout_strategy=readout,
            context=context,
            metadata=metadata or {},
        )

    @classmethod
    def choice(
        cls,
        text: str,
        choices: Sequence[Union[str, OptionDefinition]],
        question_id: Optional[str] = None,
        context: Optional[str] = None,
        descriptions: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Question:
        """Construct a categorical multiple-choice question with arbitrary number of options."""
        options: List[OptionDefinition] = []
        descriptions = descriptions or {}
        has_multi_token = False

        for item in choices:
            if isinstance(item, OptionDefinition):
                options.append(item)
                if " " in item.label:
                    has_multi_token = True
            else:
                s_item = str(item)
                if " " in s_item:
                    has_multi_token = True
                desc = descriptions.get(s_item)
                options.append(OptionDefinition(key=s_item, label=s_item, description=desc))

        qid = question_id or cls.generate_id(text, [opt.key for opt in options])
        readout = ReadoutStrategy.MULTI_TOKEN_SEQUENCE if has_multi_token else ReadoutStrategy.NEXT_TOKEN

        return cls(
            id=qid,
            text=text,
            answer_type=AnswerType.CHOICE,
            options=options,
            readout_strategy=readout,
            context=context,
            metadata=metadata or {},
        )

    @classmethod
    def ordinal(
        cls,
        text: str,
        levels: Sequence[str],
        question_id: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Question:
        """Construct an ordered/ordinal question (e.g., ['low', 'medium', 'high', 'critical'])."""
        qid = question_id or cls.generate_id(text, levels)
        options: List[OptionDefinition] = []
        has_multi_token = False

        for rank, lvl in enumerate(levels):
            if " " in lvl:
                has_multi_token = True
            options.append(OptionDefinition(key=lvl, label=lvl, ordinal_rank=rank))

        readout = ReadoutStrategy.MULTI_TOKEN_SEQUENCE if has_multi_token else ReadoutStrategy.NEXT_TOKEN
        meta = metadata or {}
        meta["ordered_levels"] = list(levels)

        return cls(
            id=qid,
            text=text,
            answer_type=AnswerType.ORDINAL,
            options=options,
            readout_strategy=readout,
            context=context,
            metadata=meta,
        )

    @classmethod
    def score(
        cls,
        text: str,
        minimum: int = 0,
        maximum: int = 10,
        step: int = 1,
        question_id: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Question:
        """Construct a numeric/scored rating question (e.g. 0 to 10)."""
        values = list(range(minimum, maximum + 1, step))
        labels = [str(v) for v in values]
        qid = question_id or cls.generate_id(text, labels)
        options = [
            OptionDefinition(key=str(v), label=str(v), numeric_value=float(v))
            for v in values
        ]
        meta = metadata or {}
        meta.update({"minimum": minimum, "maximum": maximum, "step": step})

        return cls(
            id=qid,
            text=text,
            answer_type=AnswerType.NUMERIC_SCORE,
            options=options,
            readout_strategy=ReadoutStrategy.NEXT_TOKEN,
            context=context,
            metadata=meta,
        )

    @classmethod
    def multi_choice(
        cls,
        text: str,
        choices: Sequence[str],
        question_id: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Question:
        """Construct a multi-label question where each tag can independently apply."""
        qid = question_id or cls.generate_id(text, choices)
        options = [OptionDefinition(key=str(c), label=str(c)) for c in choices]
        has_multi = any(" " in str(c) for c in choices)
        readout = ReadoutStrategy.MULTI_TOKEN_SEQUENCE if has_multi else ReadoutStrategy.NEXT_TOKEN

        return cls(
            id=qid,
            text=text,
            answer_type=AnswerType.MULTI_CHOICE,
            options=options,
            readout_strategy=readout,
            context=context,
            metadata=metadata or {},
        )

    @classmethod
    def custom(
        cls,
        text: str,
        options: Sequence[OptionDefinition],
        readout_strategy: ReadoutStrategy = ReadoutStrategy.NEXT_TOKEN,
        question_id: Optional[str] = None,
        context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Question:
        """Create a custom typed question with explicit option definitions."""
        qid = question_id or cls.generate_id(text, [opt.key for opt in options])
        return cls(
            id=qid,
            text=text,
            answer_type=AnswerType.CUSTOM,
            options=list(options),
            readout_strategy=readout_strategy,
            allow_custom_readout=True,
            context=context,
            metadata=metadata or {},
        )

