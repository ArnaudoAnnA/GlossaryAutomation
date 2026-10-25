
if __name__ == "__main__":
    """PSEUDOCODE:
        dataset_projects = load_dataset()
        
        For llm in LLMs:
        
            for project in TEST_PROJECTS:
                requirements = choose_random(project, REQUIREMENTS_PER_PROJECT)
                few_shot_dataset = dataset_projects.copy().remove(project)
                
                for n in SHOT_EXAMPLES:
                    output_file = OUTPUT_FILE_NAME.format(n=n, project=project, llm=llm)
                
                    for requirement in requirements:
                    
                        for independent_run in REPEATED_RUNS:
                            few_shots = semantic_search(few_shot_dataset, requirement, n)
                            r4_violations = llm.query(requirement, few_shots)
                            tp, tn, fp, fn = truth_matrix(r4_violations, project)
                            precision, recall, f1 = precision_recall(tp, tn, fp, fn)
                            
                            definitions = []
                            similarities = []
                            for term in tp:
                                definition = llm.query(term, project, few_shots)
                                definitions.append(definition)
                                similarity = cosine_similarity(term, project.glossary[term])
                                similarities.append(similarity)
                                
                            output_file.write(independent_run, requirement, r4_violations, precision, recall, f1, definitions, similarities)              
    """