"""
Module for evaluating the summarization system with the ROUGE metrics

To run this evaluation scipt, execute:
    python3 run_rouge_eval.py --summary_path path_to_summaries --model_path path_to_model_files
    --rouge_methods methods(separated with ',') --eval_output_path diectory_storing_eval_output
    --eval_output_filename output_filename

* All flags are required
* An example of the passing arguments to the  --rouge_methods flag:
    --rouge_methods rouge1,rouge2

"""

import os
import argparse
from sys import argv
from collections import defaultdict
from rouge_score import rouge_scorer

individual_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
total_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

def collect_summaries(summary_path: str) -> dict:
    """
    To collect all summary files and store them in a dict
    
    Parameters
    ----------
    summary_path: str
        path to the directory storing all the summary outputs
    
    Returns
    -------
    summary_dict: dict (defaultdict)
        a dict storing the summary file names
        * Format: {"summary_method_id_1": [file1, file2, ...],
                   "summary_method_id_2": [file7, file10, ...]}
    
    """
    summary_dict = defaultdict(list)
    summaries = os.listdir(summary_path)
    summaries = [summary for summary in summaries if (os.path.isfile(summary_path+"/"+summary)) and ("DS_Store" not in summary)]

    for summary in summaries:
        summary_method_id = summary.split(".")[-1]
        summary_dict[summary_method_id].append(summary)
    return summary_dict

def collect_models(model_path: str) -> dict:
    """
    To collect all the model files in a designated directory for later use

    Parameters
    ----------
    model_path: str
        path to the directory storing all the models (human summaries)
    
    Returns
    -------
    model_dict: dict
        a dict storing the eval id and the corresponding models
        * Format: {"eval_id_1": [model1, model2, ...], "eval_id_2": [model5, model6, ...]}
    
    """
    model_dict = defaultdict(list)
    models = os.listdir(model_path)

    for model in models:
        eval_id = ".".join(model.split(".")[:-1])
        model_dict[eval_id].append(model)
    return model_dict

def load_data(file_path: str) -> str:
    """
    To load data from a specified path

    Parameters
    ----------
    file_path: str
        the path of the file to be opened (str)
    
    Returns
    -------
    content: str
        the data of the specified file

    """
    try:
        with open(file_path, 'r', encoding = "utf8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding = "cp1252") as f:
            content = f.read()
    return content

def evaluate_results(summary_dict: dict, model_dict: dict):
    """
    Run system evaluation on ROUGE metrics
    
    Parameters
    ----------
    summary_dict: dict (defaultdict)
        a dict storing the summary file names
        * Format: {"summary_method_id_1": [file1, file2, ...],
                   "summary_method_id_2": [file7, file10, ...]}
    model_dict: dict
        a dict storing the eval id and the corresponding models
        * Format: {"eval_id_1": [model1, model2, ...], "eval_id_2": [model5, model6, ...]}
    
    Returns
    -------
    Nothing is returned

    """
    rouge_methods = args.rouge_methods
    for summary_method_id, summary_files in summary_dict.items():
        for summary_file in summary_files:
            model_eval_id = ".".join(summary_file.split(".")[:-1])
            summary = load_data(args.summary_path + "/" + summary_file)
            summary_rouge_scores = defaultdict(lambda: defaultdict(list)) # store the results of the current summary file before taking the average (scores evaluated on different model files)
            for model_file in model_dict[model_eval_id]:
                model = load_data(args.model_path + "/" + model_file)
                scorer = rouge_scorer.RougeScorer(rouge_methods, use_stemmer=True)
                scores = scorer.score(model, summary)

                for rouge_method in rouge_methods:
                    precision, recall, fmeasure = scores[rouge_method]
                    summary_rouge_scores[rouge_method]["P"].append(precision)
                    summary_rouge_scores[rouge_method]["R"].append(recall)
                    summary_rouge_scores[rouge_method]["F"].append(fmeasure)
            
            for rouge_method, score_type_results in summary_rouge_scores.items():
                summary_all_p = score_type_results["P"]
                summary_all_r = score_type_results["R"]
                summary_all_f = score_type_results["F"]
                avg_p = round(sum(summary_all_p)/len(summary_all_p), 5)
                avg_r = round(sum(summary_all_r)/len(summary_all_r), 5)
                avg_f = round(sum(summary_all_f)/len(summary_all_f), 5)

                individual_results[summary_method_id][rouge_method][summary_file]["P"] = avg_p
                individual_results[summary_method_id][rouge_method][summary_file]["R"] = avg_r
                individual_results[summary_method_id][rouge_method][summary_file]["F"] = avg_f
                
                total_results[summary_method_id][rouge_method]["P"].append(avg_p)
                total_results[summary_method_id][rouge_method]["R"].append(avg_r)
                total_results[summary_method_id][rouge_method]["F"].append(avg_f)

    return

def output_eval_results():
    """
    Print out the evaluation results to a file
    
    Parameters
    ----------
    No parameters

    Returns
    -------
    Nothing is returned
    
    """
    eval_output = ""
    for summary_method_id, all_rouge_scores in total_results.items():
        for rouge_score_type, rouge_scores in all_rouge_scores.items():
            # get average of all summaries
            total_avg_r = round(sum(rouge_scores["R"])/len(rouge_scores["R"]), 5)
            total_avg_p = round(sum(rouge_scores["P"])/len(rouge_scores["P"]), 5)
            total_avg_f = round(sum(rouge_scores["F"])/len(rouge_scores["F"]), 5)

            # get eval output content
            eval_output += "---------------------------------------------\n"
            eval_output += "%s ROUGE-%s Average_R: %.5f\n"%(summary_method_id, rouge_score_type[-1], total_avg_r)
            eval_output += "%s ROUGE-%s Average_P: %.5f\n"%(summary_method_id, rouge_score_type[-1], total_avg_p)
            eval_output += "%s ROUGE-%s Average_F: %.5f\n"%(summary_method_id, rouge_score_type[-1], total_avg_f)
            eval_output += ".............................................\n"

            for summary_filename, summary_results in individual_results[summary_method_id][rouge_score_type].items():
                eval_output += "%s ROUGE-%s Eval %s R:%.5f P:%.5f F:%.5f\n"%(summary_method_id, rouge_score_type[-1], summary_filename, summary_results["R"], summary_results["P"], summary_results["F"])
    
    # check if directory exists, if not, create one
    if not os.path.exists(args.eval_output_path):
        os.makedirs(args.eval_output_path)
    
    # output summary
    with open(os.path.join(args.eval_output_path, args.eval_output_filename), "w") as f:
        f.write(eval_output)

    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run ROUGE evaluation metrics for summarization system.")
    
    parser.add_argument(
        "--summary_path", type=str, required=True, help="path to summmarization system outputs")
    parser.add_argument(
        "--model_path", type=str, required=True, help="path to models (gold standard)")
    parser.add_argument(
        "--rouge_methods", required=True, help="ROUGE methods used for evaluation (separated with comma)", 
        type=lambda s: [item for item in s.split(',')])
    parser.add_argument(
        "--eval_output_path", type=str, required=True, help="path to evaluation output")
    parser.add_argument(
        "--eval_output_filename", type=str, required=True, help="filename of evaluation output")
    
    args = parser.parse_args(argv[1:])
    
    # organize data into format that can be easily used
    summary_dict = collect_summaries(args.summary_path)
    model_dict = collect_models(args.model_path)

    # run evaluation metrics
    evaluate_results(summary_dict, model_dict)

    # sort the method ids in the result dict
    summary_method_ids = list(total_results.keys())
    summary_method_ids.sort()
    total_results = {i: total_results[i] for i in summary_method_ids}

    # print out results to a file
    output_eval_results()