from pathlib import Path

from LLM.adapters.interface import LLMProvider
from dataset.index import DatasetProject

ROOT_DIR = Path(__file__).resolve().parent

MAX_ATTEMPTS = 5
"""The number of attempts performed when a LLM query fails (e.g., due to network errors)."""
REPEATED_RUNS = 3
"""For each configuration, this number of independent runs is executed. The results are averaged to achieve the final output."""
OUTPUT_DIR = "output"
OUTPUT_FILE_NAME = "{llm}_{shots}-shots_{project}.csv"
"""Each file contains all the independent runs executed for the same configuration."""
OUTPUT_FILE_HEADER = ["Run", "Input_requirement", "Identified_violations", "Precision", "Recall", "F1-score", "Generated_definitions", "Cosine_similarities"]
FAILED_RUNS_FILE_NAME = "failed_runs.txt"

LLMs = [
    ("gemini-3.8-flash", LLMProvider.GOOGLE),
    ("Qwen/Qwen2.5-14B-Instruct", LLMProvider.HUGGINGFACE),
    ("meta-llama/Llama-3.1-8B-Instruct", LLMProvider.HUGGINGFACE)
]
TEMPERATURE = 0.0
SHOT_EXAMPLES = [0, 5, 15]
SYS_PROMPT = """INCOSE rule 4 (defined terms) states:

As systems become increasingly complex and software intensive, the ability to share data and 
information, including needs and requirements, across organizations both internal and external, is 
critical to project success.  The definition and documentation of a common ontology by 
organizations and their projects is fundamental.  
This ontology includes the formal naming and definition of a set of terms, entities, data 
types, and properties as well as defining the relationships among these terms, entities, and data 
types that are fundamental to the project and organization of which the project is part.   
Having a documented, common ontology and an associated glossary and/or data dictionary, 
helps ensure consistent use of this data and information across all system lifecycle process 
activities and work products as well as across various groups within and external to the project.  
This common ontology is key to tool interoperability and the ability to share sets of data and 
information, including needs and requirements.  
Most natural languages are rich with words having several synonyms, each with a subtly different 
meaning based on context. 
In need statements and requirement statements, shades of meaning will most likely lead to 
ambiguity and to difficulty during verification and validation activities across the lifecycle. 
This rule mandated to define 
terms in some form of ontology, including a glossary, data dictionary, or similar artifact that allows 
the reader of a need statement or requirement statement to know exactly what the writer’s 
intended meaning was when the word was chosen.
"""
USER_PROMPT = ("Find the violations to INCOSE R4. "
                "Just report (verbatim) the strings that are related to a violation. "
                "The strings should be separated by commas. "
                "Find the violations in this requirement: {requirement_text}")

TEST_PROJECTS = [DatasetProject.NASA, DatasetProject.INVENTORY, DatasetProject.SCEAPI]
REQUIREMENTS_PER_PROJECT = 5
"""The number of requirements randomly extracted for each project in the test dataset."""
COLLECTION_NAME: str = "few_shot_examples"
"""The name of the collection used for RAG."""
