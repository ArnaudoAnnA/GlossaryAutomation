import json
import random
from enum import Enum
from pathlib import Path

from RAG.models import RequirementEntry, TermReplacement

DATASET_DIR = Path(__file__).parent.resolve()
GLOSSARY_FILE_NAME = "{project}_glossary.csv"
TEST_DATA_FILE_NAME = "{project}_test-data.json"


class DatasetProject(Enum):
    ELSFORK = "elsfork"
    GAMMA_J = "gamma-j"
    GEMINI = "gemini"
    HOME = "home"
    INVENTORY = "inventory"
    NASA = "nasa"
    PEPPOL = "peppol"
    SCEAPI = "sceapi"


ALL_PROJECTS = [v for v in DatasetProject]

def get_project_path(project_name: DatasetProject):
    return DATASET_DIR / Path(TEST_DATA_FILE_NAME.format(project=project_name.value))

def load_requirements(project_name: DatasetProject) -> list[RequirementEntry]:
    path = get_project_path(project_name)
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        RequirementEntry(
            requirement_text=item["requirement_text"],
            r4_violations=[TermReplacement(**v) for v in item["r4_violations"]]
        )
        for item in data
    ]

def get_random_requirements(project_name: DatasetProject, quantity: int) -> list[RequirementEntry]:
    requirements = load_requirements(project_name)
    sample_size = min(quantity, len(requirements))
    return random.sample(requirements, k=sample_size)

