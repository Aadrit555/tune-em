"""Prompt template abstraction and built-in ensemble templates."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from anydecision.core.question import Question


class PromptTemplate(BaseModel):
    """Encapsulates a prompt layout for eliciting decisions directly from logits."""
    name: str = Field(description="Unique name of this template.")
    system: Optional[str] = Field(default=None, description="Optional system prompt.")
    prefix_template: str = Field(
        default="Question: {question}\nOptions:\n{options}\nAnswer:",
        description="Format string containing {question} and {options} placeholders."
    )
    option_format: str = Field(
        default="- {label}",
        description="Format string for each candidate option ({key}, {label}, {index})."
    )
    separator: str = Field(
        default="\n",
        description="Separator between rendered options."
    )

    def render(self, question: Question, option_order: Optional[List[str]] = None) -> str:
        """Render question and ordered options into full prompt text."""
        order = option_order or question.option_keys()
        formatted_options = []
        for idx, key in enumerate(order):
            opt = question.get_option_by_key(key)
            label = opt.label if opt else key
            desc = f" ({opt.description})" if (opt and opt.description) else ""
            line = self.option_format.format(
                key=key,
                label=f"{label}{desc}",
                index=idx + 1,
                letter=chr(65 + idx) if idx < 26 else str(idx + 1)
            )
            formatted_options.append(line)

        options_str = self.separator.join(formatted_options)
        text = question.text
        if question.context:
            text = f"Context: {question.context}\n\n{text}"

        rendered = self.prefix_template.format(question=text, options=options_str)
        if self.system:
            rendered = f"{self.system}\n\n{rendered}"
        return rendered


class TemplateRegistry:
    """Registry containing standard and user-defined prompt templates."""

    def __init__(self) -> None:
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(PromptTemplate(
            name="minimal",
            prefix_template="Question: {question}\nOptions: {options}\nDecision:",
            option_format="{label}",
            separator=", ",
        ))
        self.register(PromptTemplate(
            name="structured",
            prefix_template="Task: Evaluate the question and select the exact best matching answer.\n\nQuestion: {question}\n\nAllowed Choices:\n{options}\n\nSelected Answer:",
            option_format="* {label}",
            separator="\n",
        ))
        self.register(PromptTemplate(
            name="qa",
            prefix_template="Q: {question}\n\nCandidate Answers:\n{options}\n\nA:",
            option_format="({letter}) {label}",
            separator="\n",
        ))
        self.register(PromptTemplate(
            name="cloze",
            prefix_template="Regarding the statement: \"{question}\"\nThe correct specification among [{options}] is:",
            option_format="\"{label}\"",
            separator=", ",
        ))
        self.register(PromptTemplate(
            name="deliberative",
            prefix_template="Carefully consider the question below and determine the definitive classification.\n\nInput: {question}\nOptions:\n{options}\n\nClassification:",
            option_format="- [{letter}] {label}",
            separator="\n",
        ))

    def register(self, template: PromptTemplate) -> None:
        """Register a new template."""
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        """Retrieve a template by name."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[name]

    def list_templates(self) -> List[str]:
        """List all registered template names."""
        return list(self._templates.keys())


DEFAULT_TEMPLATES = TemplateRegistry()
