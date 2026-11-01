

class Evaluation():
    """
    Run recall@k in search results against a dataset path.
    Params:
        - `student_search_results_path`: The program generated
        results to evaluate
        - `dataset_path`: The supervised dataset of relevant 
        items (AnwseredQuestions)
    
    We run recall@k for k = {1, 3, 5, 10}
    For this we have to evaluate the top-k sources in 
    each MinimalSearchResult question_id in each one.
    Once we have the MinimalSource (file_path,  first_character_index,
    last_character_index)
        - We now check if the source file_path matches
        - If they match, do they math first char and last_char? Leaving some
        range.
        - If they dont, then it doesnt count.
        - If they do. We add it to the score
    """
    