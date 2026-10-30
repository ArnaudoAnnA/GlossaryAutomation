import csv
import traceback
from pathlib import Path

from LLM.adapters.interface import get_llm_adapter
from RAG.initialize_rag import get_rag_manager
from constants import TEST_PROJECTS, LLMs, COLLECTION_NAME, SYS_PROMPT, REQUIREMENTS_PER_PROJECT, SHOT_EXAMPLES, REPEATED_RUNS, OUTPUT_FILE_NAME, OUTPUT_DIR, OUTPUT_FILE_HEADER, USER_PROMPT_VIOLATIONS, FAILED_RUNS_FILE_NAME
from dataset.index import get_project_path, ALL_PROJECTS, get_random_requirements


def compute_metrics(r4_violations_generated: set[str], r4_violations_ground_truth: set[str]) -> tuple[float, float, float, list[str], list[str], list[str]]:
    r4_violations_generated = set([v.strip() for v in r4_violations_generated])

    tp = r4_violations_generated & r4_violations_ground_truth
    fp = r4_violations_generated - r4_violations_ground_truth
    fn = r4_violations_ground_truth - r4_violations_generated

    tp_count = len(tp)
    fp_count = len(fp)
    fn_count = len(fn)

    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return prec, rec, f1, tp, fp, fn



if __name__ == "__main__":
    """PSEUDOCODE:
        for project in TEST_PROJECTS:
            few_shot_projects = ALL_PROJECTS.copy().remove(project)
            few_shot_dataset = load_dataset(few_shot_projects)  
            requirements = choose_random(project, REQUIREMENTS_PER_PROJECT)
            
            For llm_name in LLMs:
                
                for n in SHOT_EXAMPLES:
                    output_file = OUTPUT_FILE_NAME.format(n=n, project=project, llm=llm)
                
                    for requirement in requirements:
                    
                        for independent_run in REPEATED_RUNS:
                            llm = init_llm()
                            few_shots = semantic_search(few_shot_dataset, requirement, n)
                            r4_violations = llm.query(requirement, few_shots)
                            tp, tn, fp, fn = truth_matrix(r4_violations, project)
                            precision, recall, f1 = precision_recall(tp, tn, fp, fn)
                            
                            definitions = []
                            similarities = []
                            for term in tp:
                                match = get_match(term, project)
                                definition = llm.query(term, project, few_shots)
                                definitions.append(definition)
                                similarity = cosine_similarity(term, project.glossary[match])
                                similarities.append(similarity)
                                
                            output_file.write(independent_run, requirement, r4_violations, precision, recall, f1, definitions, similarities)    
            
            clear_dataset()          
    """
    from huggingface_hub import login
    login(token="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")

    rag_manager = get_rag_manager()

    with open(Path(OUTPUT_DIR) / FAILED_RUNS_FILE_NAME, "w") as file:
        file.write("")

    for project_name in TEST_PROJECTS:
        print(project_name)
        # Few-shot data is all the dataset except the current test project
        rag_manager.clear_collection(COLLECTION_NAME)
        few_shot_projects = ALL_PROJECTS.copy()
        few_shot_projects.remove(project_name)
        rag_manager.load_data(collection_name=COLLECTION_NAME, projects=few_shot_projects)

        # Test data
        test_requirements = get_random_requirements(project_name=project_name, quantity=REQUIREMENTS_PER_PROJECT)

        for llm_info in LLMs:
            print(f"\t{llm_info[0]}")

            for n_examples in SHOT_EXAMPLES:
                print(f"\t\t{n_examples}-shot")
                output_file = Path(OUTPUT_DIR) / OUTPUT_FILE_NAME.format(shots=n_examples, project=project_name, llm=llm_info[0].replace("/", "_"))

                if output_file.exists():
                    print(f"\t\t\t{output_file} already exists => skipping")
                    continue # Skip this configuration

                with open(output_file, "w") as file:
                    writer = csv.writer(file, delimiter=";")
                    writer.writerow(OUTPUT_FILE_HEADER)

                for requirement in test_requirements:

                    for independent_run in range(REPEATED_RUNS):
                        try:
                            # --- SETUP --------------------------------------------------------------------------------
                            few_shot_examples = rag_manager.retrieve_similar_texts(collection_name=COLLECTION_NAME,
                                                                                   text=requirement.requirement_text,
                                                                                   top_k=n_examples)
                            formatted_shot_examples = [se.to_few_shot_example_for_violation_identification() for se in few_shot_examples]

                            # llm instances are not recycled: for each experiment, a new instance is generated with empty chat
                            llm = get_llm_adapter(model_id=llm_info[0], sys_prompt=SYS_PROMPT, model_provider=llm_info[1], shot_examples=formatted_shot_examples)
                            # -------------------------------------------------------------------------------------------

                            # STEP 1: violations identification
                            r4_violations: list[str] = llm.make_a_query(USER_PROMPT_VIOLATIONS.format(requirement_text=requirement.requirement_text)).split(",")
                            precision, recall, f1_score, tp, fp, fn = compute_metrics(set(r4_violations), set([t.term_to_substitute for t in requirement.r4_violations]))

                            # STEP 2: definitions generations
                            definitions = []
                            for item in tp:
                                definition = llm.make_a_query()

                            with open(output_file, "a") as file:
                                writer = csv.writer(file, delimiter=";")
                                writer.writerow([independent_run, requirement.__str__(), r4_violations, precision, recall, f1_score, tp, fp, fn, "", ""])
                                file.flush()


                        except Exception as e:
                            failed_run = (f"(project_name = {project_name}, "
                                  f"llm_name = {llm_info[0]},  "
                                  f"n_examples = {n_examples}, "
                                  f"requirement = {requirement.requirement_text}, "
                                  f"independent_run = {independent_run})\n")
                            print(f"ERROR: Impossible to obtain results for this configuration: {failed_run}\n")
                            with open(Path(OUTPUT_DIR) / FAILED_RUNS_FILE_NAME, "a") as file:
                                file.write(failed_run)
                                file.flush()
                            traceback.print_exc()













