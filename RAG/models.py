from dataclasses import dataclass


@dataclass
class GlossaryEntry:
    """Represents a glossary entry without a relevance score."""

    term: str
    definition: str


@dataclass
class GlossaryMatch:
    """Represents a glossary entry retrieved from a vector search with relevance score."""

    term: str
    definition: str
    score: float

    def __str__(self) -> str:
        return f'{self.term} (confidence {self.score}): _{self.definition}_ '
