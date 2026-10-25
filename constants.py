MAX_ATTEMPTS = 5



LLMs = ["gemini-3.8-flash", "Qwen/Qwen2.5-32B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"]
TEMPERATURE = 0.0
SHOT_EXAMPLES = [0, 5, 15]
REPEATED_RUNS = 3
"""For each configuration, this number of independent runs is executed. The results are averaged to achieve the final output."""


TEST_PROJECT = ["NASA", "Inventory", "Sce api"]
REQUIREMENTS_PER_PROJECT = 5
"""The number of requirements randomly extracted for each project in the test dataset."""