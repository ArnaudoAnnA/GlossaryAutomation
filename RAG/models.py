from dataclasses import dataclass


@dataclass
class GlossaryEntry:
    """Represents a glossary entry without a relevance score."""

    term: str
    definition: str

@dataclass
class TermReplacement:
    term_to_substitute: str
    target: str

    def __str__(self) -> str:
        return f'{self.term_to_substitute} -> {self.target}'

@dataclass
class RequirementEntry:
    requirement_text: str
    r4_violations: list[TermReplacement]

    def __str__(self) -> str:
        violations_str = "\n".join([v.__str__() for v in self.r4_violations])
        return f'{self.requirement_text}\n\n{violations_str}'

    def to_few_shot_example_for_violation_identification(self) -> tuple[str, str]:
        return (f"Find the violations to INCOSE R4 in this requirement: {self.requirement_text}",
                ', '.join([t.term_to_substitute for t in self.r4_violations]))


@dataclass
class GlossaryMatch:
    """Represents a glossary entry retrieved from a vector search with relevance score."""

    term: str
    definition: str
    score: float

    def __str__(self) -> str:
        return f'{self.term} (confidence {self.score}): _{self.definition}_ '
