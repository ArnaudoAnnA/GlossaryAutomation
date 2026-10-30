import csv
import traceback
from pathlib import Path

from constants import OUTPUT_DIR, FAILED_RUNS_FILE_NAME, TEST_PROJECTS, LLMs, SHOT_EXAMPLES, OUTPUT_FILE_NAME, OUTPUT_FILE_HEADER

if __name__ == "__main__":
    with open(Path(OUTPUT_DIR) / FAILED_RUNS_FILE_NAME, "w") as file:
        file.write("")

    for project_name in TEST_PROJECTS:
        print(project_name)

        for llm_info in LLMs:
            print(f"\t{llm_info[0]}")

            for n_examples in SHOT_EXAMPLES:
                try:
                    print(f"\t\t{n_examples}-shot")
                    output_file = Path(OUTPUT_DIR) / OUTPUT_FILE_NAME.format(shots=n_examples, project=project_name, llm=llm_info[0].replace("/", "_"))

                    precision_tot, recall_tot, count = 0.0, 0.0, 0
                    index_precision = OUTPUT_FILE_HEADER.index("Precision")
                    index_recall = OUTPUT_FILE_HEADER.index("Recall")
                    index_f1 = OUTPUT_FILE_HEADER.index("F1-score")

                    with open(output_file, "r") as file:
                        reader = csv.reader(file, delimiter=";")
                        next(reader, None)  # Skip the header safely

                        for line in reader:
                            if line:  # Avoid empty lines
                                precision_tot += float(line[index_precision])
                                recall_tot += float(line[index_recall])
                                count += 1

                    avg_precision = precision_tot / count
                    avg_recall = recall_tot / count

                    if (avg_precision + avg_recall) > 0:
                        avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
                    else:
                        avg_f1 = 0.0

                    line = ["" for _ in OUTPUT_FILE_HEADER]
                    line[index_precision] = str(avg_precision)
                    line[index_recall] = str(avg_recall)
                    line[index_f1] = str(avg_f1)
                    line[0] = "Averages"

                    with open(output_file, "a", newline="") as file:
                        writer = csv.writer(file, delimiter=";")
                        writer.writerow(line)
                        file.flush()

                except Exception as e:
                    failed_run = (f"(project_name = {project_name}, "
                                  f"llm_name = {llm_info[0]},  "
                                  f"n_examples = {n_examples})\n")
                    print(f"ERROR: Impossible to obtain results for this configuration: {failed_run}\n")
                    with open(Path(OUTPUT_DIR) / FAILED_RUNS_FILE_NAME, "a") as file:
                        file.write(failed_run)
                        file.flush()
                    traceback.print_exc()