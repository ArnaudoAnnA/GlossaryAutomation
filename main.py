import traceback

from LLM.adapters.interface import get_llm_adapter
from RAG.initialize_rag import get_rag_manager
from constants import TEST_PROJECTS, LLMs, COLLECTION_NAME, SYS_PROMPT, REQUIREMENTS_PER_PROJECT, SHOT_EXAMPLES, REPEATED_RUNS, OUTPUT_FILE_NAME
from dataset.index import get_dataset_path, ALL_PROJECTS, get_random_requirements


def compute_metrics(r4_violations_generated: set[str], r4_violations_ground_truth: set[str]) -> tuple[float, float, float]:
    tp = len(r4_violations_generated & r4_violations_ground_truth)
    fp = len(r4_violations_generated - r4_violations_ground_truth)
    fn = len(r4_violations_ground_truth - r4_violations_generated)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return prec, rec, f1



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

    rag_manager = get_rag_manager()


    for project_name in TEST_PROJECTS:
        # Few-shot data is all the dataset except the current test project
        few_shot_projects = ALL_PROJECTS.copy()
        few_shot_projects.remove(project_name)
        few_shot_projects_paths = [get_dataset_path(project_name=p) for p in few_shot_projects]
        rag_manager.load_data(collection_name=COLLECTION_NAME, paths=few_shot_projects_paths)

        # Test data
        test_requirements = get_random_requirements(project_name=project_name, quantity=REQUIREMENTS_PER_PROJECT)

        for llm_info in LLMs:

            for n_examples in SHOT_EXAMPLES:
                output_file = OUTPUT_FILE_NAME.format(n=n_examples, project=project_name, llm=llm_info[0])

                for requirement in test_requirements:

                    for independent_run in range(REPEATED_RUNS):
                        try:
                            # --- SETUP --------------------------------------------------------------------------------
                            few_shot_examples = rag_manager.retrieve_similar_texts(collection_name=COLLECTION_NAME,
                                                                                   text=requirement.requirement_text,
                                                                                   field_name="requirement_text",
                                                                                   top_k=n_examples)
                            formatted_shot_examples = [se.to_few_shot_example_for_violation_identification() for se in few_shot_examples]

                            # llm instances are not recycled: for each experiment, a new instance is generated with empty chat
                            llm = get_llm_adapter(model_id=llm_info[0], sys_prompt=SYS_PROMPT, model_provider=llm_info[1], shot_examples=formatted_shot_examples)
                            # -------------------------------------------------------------------------------------------

                            # STEP 1: violations identification
                            r4_violations: list[str] = llm.make_a_query(f"Find the violations to INCOSE R4 in this requirement: {requirement.requirement_text}").split(",")
                            precision, recall, f1_score = compute_metrics(set(r4_violations), set([t.target for t in requirement.r4_violations]))

                            # STEP 2: definitions generations
                            # TODO


                        except Exception as e:
                            print(f"ERROR: Impossible to obtain results for this configuration: "
                                  f"(project_name = {project_name}, "
                                  f"llm_name = {llm_info[0]},  "
                                  f"n_examples = {n_examples}, "
                                  f"requirement = {requirement}, "
                                  f"independent_run = {independent_run}) \n")
                            traceback.print_exc()




        rag_manager.clear_dataset(COLLECTION_NAME)






