from dataset.index import DatasetProject

MAX_ATTEMPTS = 5
"""The number of attempts performed when a LLM query fails (e.g., due to network errors)."""
REPEATED_RUNS = 3
"""For each configuration, this number of independent runs is executed. The results are averaged to achieve the final output."""
OUTPUT_DIR = "output"
OUTPUT_FILE_NAME = "{llm}_{shots}-shots_{project}.csv"
"""Each file contains all the independent runs executed for the same configuration."""
OUTPUT_FILE_HEADER = ["Run", "Input_requirement", "Identified_violations", "Precision", "Recall", "F1-score", "Generated_definitions", "Cosine_similarities"]

LLMs = ["gemini-3.8-flash", "Qwen/Qwen2.5-32B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"]
TEMPERATURE = 0.0
SHOT_EXAMPLES = [0, 5, 15]

TEST_PROJECT = [DatasetProject.NASA, DatasetProject.INVENTORY, DatasetProject.SCEAPI]
REQUIREMENTS_PER_PROJECT = 5
"""The number of requirements randomly extracted for each project in the test dataset."""