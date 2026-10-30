import csv
import traceback
from operator import index
from pathlib import Path

from constants import OUTPUT_DIR, FAILED_RUNS_FILE_NAME, TEST_PROJECTS, LLMs, SHOT_EXAMPLES, OUTPUT_FILE_NAME, OUTPUT_FILE_HEADER

AGGREGATED_RESULTS_FILE_NAME = "aggregated_results.csv"
AGGREGATED_RESULTS_HEADER = ["Project", "LLM", "N_EXAMPLES", "Precision", "Recall", "F1-score"]

if __name__ == "__main__":
    with open(Path(OUTPUT_DIR) / FAILED_RUNS_FILE_NAME, "w") as file:
        file.write("")

    with open(Path(OUTPUT_DIR) / AGGREGATED_RESULTS_FILE_NAME, "w") as out_file:
        writer = csv.writer(out_file, delimiter=";")
        writer.writerow(AGGREGATED_RESULTS_HEADER)

        for project_name in TEST_PROJECTS:
            print(project_name)

            for llm_info in LLMs:
                print(f"\t{llm_info[0]}")

                for n_examples in SHOT_EXAMPLES:
                    try:
                        print(f"\t\t{n_examples}-shot")
                        input_file = Path(OUTPUT_DIR) / OUTPUT_FILE_NAME.format(shots=n_examples, project=project_name, llm=llm_info[0].replace("/", "_"))

                        last_line = None
                        with open(input_file, "r") as file:
                            reader = csv.reader(file, delimiter=";")
                            next(reader, None)  # Skip the header safely

                            for line in reader:
                                if line:  # Avoid empty lines
                                    last_line = line

                        index_precision = OUTPUT_FILE_HEADER.index("Precision")
                        index_recall = OUTPUT_FILE_HEADER.index("Recall")
                        index_f1 = OUTPUT_FILE_HEADER.index("F1-score")

                        precision = last_line[index_precision]
                        recall = last_line[index_recall]
                        f1 = last_line[index_f1]

                        output_line = [project_name, llm_info[0], n_examples, precision, recall, f1]
                        writer.writerow(output_line)
                        out_file.flush()

                    except Exception as e:
                        failed_run = (f"(project_name = {project_name}, "
                                      f"llm_name = {llm_info[0]},  "
                                      f"n_examples = {n_examples})\n")
                        print(f"ERROR: Impossible to obtain results for this configuration: {failed_run}\n")
                        with open(Path(OUTPUT_DIR) / FAILED_RUNS_FILE_NAME, "a") as file:
                            file.write(failed_run)
                            file.flush()
                        traceback.print_exc()