import argparse
import os
import json
from tqdm import tqdm

from datasets import load_dataset

from src.utils import map_to_instruct, map_from_instruct_only_action, remove_thought_in_dpo

ques_cfg = {}

ques_cfg['no_ques'] = {
        'question_adv_thresh': 100.00,
        'priv_action_gap_max': 0.0,
        'priv_action_gap_negligible': 100.0
}

ques_cfg['normal'] = {
    'question_adv_thresh': 0.2,
    'priv_action_gap_max': 0.5,
    'priv_action_gap_negligible': 0.0,
    }


def step_in_data(data):
    return data['xs'].count('<|start_header_id|>assistant<|end_header_id|>')-1

def too_many_failures(data):
    remove_failures_over = 2
    failures_list = [p.startswith('Action Failed!') for p in data['xs'].split('start_header_id|>environment<|end_header_id|>\n\nObservation: ')]
    if len(failures_list) < remove_failures_over:
        return False
    else:
        return all(failures_list[-remove_failures_over:])

def compute_ask_question(data, no_questions=False):
    p = data['probabilities']
    if p is None:
        return 'None'
    if too_many_failures(data):
        return 'None'
    if p['p_s_a_t'] > p['p_s_a_s'] - ques_cfg['no_ques' if no_questions else 'normal']['priv_action_gap_negligible']:
        return 'at'
    # If question helps the student predict 'at', ask the question
    elif p['p_s_a_t_ques'] > p['p_s_a_t'] + ques_cfg['no_ques' if no_questions else 'normal']['question_adv_thresh']:
        return 'ques'
    # If student is reasonably able to predict 'at', directly predict 'at'
    elif p['p_s_a_t'] > p['p_s_a_s'] - ques_cfg['no_ques' if no_questions else 'normal']['priv_action_gap_max']:
        return 'at'
    # If the student cannot predict 'at', even after the question, then maybe 'as' was okay.
    return 'as'


def regenerate_data(data_path, no_questions=False):
    data_filename_sft_corr = os.path.join(data_path, "gen_rawdata_sft_corr.jsonl")
    data_filename_sft_basic = os.path.join(data_path, "gen_rawdata_sft_base.jsonl")
    data_filename_sft_all = os.path.join(data_path, "gen_rawdata_sft_everything.jsonl")
    data_filename_dpo_corr = os.path.join(data_path, "gen_rawdata_dpo_corr.jsonl")
    open(data_filename_sft_corr, "w").write("")
    open(data_filename_sft_basic, "w").write("")
    open(data_filename_sft_all, "w").write("")
    open(data_filename_dpo_corr, "w").write("")
    old_dataset_flag=False
    
    print("Generating data for SFT and DPO....")
    if os.path.exists(os.path.join(data_path, "data_with_labels_and_probs.json")):
        base_dataset = load_dataset("json", data_files=os.path.join(data_path, "data_with_labels_and_probs.json"))
        stats = {'as':0, 'at':0, 'ques':0}
        for datum in tqdm(base_dataset['train']):
            if datum['as'].split('Action: ')[-1] == datum['at'].split('Action: ')[-1]:
                todo = 'as'
            else:
                todo = compute_ask_question(datum, no_questions=no_questions)
            sft_data = {'prompt_priv':datum['xt'],
                            'prompt_unpriv':datum['xs'],
                            'completion':datum['as']+'<|eot_id|>'}
            open(data_filename_sft_basic, "a").write(json.dumps(sft_data) + "\n")
            open(data_filename_sft_all, "a").write(json.dumps(sft_data) + "\n")
            if todo == 'as':
                stats['as'] += 1
                open(data_filename_sft_corr, "a").write(json.dumps(sft_data) + "\n")


            sft_data = {'prompt_priv':datum['xt'],
                            'prompt_unpriv':datum['xs'],
                            'completion':datum['at']+'<|eot_id|>'}
            dpo_data = {'prompt_priv':datum['xt'],
                            'prompt_unpriv':datum['xs'],
                            'chosen':datum['at']+'<|eot_id|>',
                            'rejected':datum['as']+'<|eot_id|>'}
            open(data_filename_sft_all, "a").write(json.dumps(sft_data) + "\n")
            if todo == 'at':
                stats['at'] += 1
                open(data_filename_dpo_corr, "a").write(json.dumps(dpo_data) + "\n")
                open(data_filename_sft_corr, "a").write(json.dumps(sft_data) + "\n")


            if datum['ques'] is None:
                continue
            
            sft_data = {'prompt_priv':datum['xt'],
                            'prompt_unpriv':datum['xs'],
                            'completion':datum['ques']+'<|eot_id|>'}
            dpo_data = {'prompt_priv':datum['xt'],
                            'prompt_unpriv':datum['xs'],
                            'chosen':datum['ques']+'<|eot_id|>',
                            'rejected':datum['as']+'<|eot_id|>'}
            open(data_filename_sft_all, "a").write(json.dumps(sft_data) + "\n")
            if todo == 'ques':
                stats['ques'] += 1
                open(data_filename_dpo_corr, "a").write(json.dumps(dpo_data) + "\n")
                open(data_filename_sft_corr, "a").write(json.dumps(sft_data) + "\n")

    else:
        old_dataset_flag=True
        print("data_with_labels_and_probs.json not found. Looks like an older dataset....")
        assert os.path.exists(os.path.join(data_path, "online_train_set_SFT_planner.jsonl")), "online_train_set_SFT_planner.jsonl not found. Cannot determine dataset type."
        assert os.path.exists(os.path.join(data_path, "data_labels_all.json")), "data_labels_all.json not found. Cannot determine dataset type."

        data_filename_sft_corr = os.path.join(data_path, "data_labels_all.json")
        data_filename_sft_all = os.path.join(data_path, "data_labels_all.json")
        data_filename_sft_basic_text = os.path.join(data_path, "online_train_set_SFT_planner.jsonl")

        datasets = load_dataset("json", data_files=data_filename_sft_basic_text)["train"]
        datasets = datasets.map(map_to_instruct)
        data_filename_sft_basic = os.path.join(data_path, "data_labels_completion_basic.json")
        datasets.to_json(data_filename_sft_basic)

        data_filename_dpo_corr = os.path.join(data_path, "data_dpo.json")

    open(os.path.join(data_path, "config.json"), "w").write(json.dumps({'config':ques_cfg['no_ques' if no_questions else 'normal'],'stats':stats} , indent=4))

    prompt_columns = ['prompt_priv', 'prompt_unpriv']

    for dataset_name, dataset_path in zip(['SFT_correction'], [data_filename_sft_corr]):
        print(f"\n\nGenerating data for {dataset_name}....")
        datasets = load_dataset("json", data_files=dataset_path)["train"].train_test_split(test_size=0.1)
        for split, dataset in datasets.items():
            if not old_dataset_flag:
                dataset_local = dataset.rename_column(f'prompt_unpriv_verbose', 'prompt').remove_columns([p for p in prompt_columns if p != f'prompt_unpriv_verbose'])
            else:
                dataset_local = dataset
            dataset_local_only_action = dataset_local.map(map_from_instruct_only_action, batched=True)
            dataset_local_only_action.to_json(os.path.join(data_path, f"gen_dataset_{dataset_name}_unpriv_verbose_{split}_onlyAction.jsonl"))

    ## DPO data for Planner Correction
    print(f"\n\nGenerating data for DPO....")
    datasets = load_dataset("json", data_files=data_filename_dpo_corr)["train"].train_test_split(test_size=0.1)
    for split, dataset in datasets.items():
        if not old_dataset_flag:
            dataset_local = dataset.rename_column(f'prompt_unpriv_verbose', 'prompt').remove_columns([p for p in prompt_columns if p != f'prompt_unpriv_verbose'])
        else:
            dataset_local = dataset
        dataset_local_only_action = dataset_local.map(remove_thought_in_dpo, batched=True)
        dataset_local_only_action.to_json(os.path.join(data_path, f"gen_dataset_DPO_unpriv_verbose_{split}_onlyAction.jsonl"))

    os.remove(data_filename_sft_corr)
    os.remove(data_filename_sft_basic)
    os.remove(data_filename_sft_all)
    os.remove(data_filename_dpo_corr)

    print(f"\n\n\n\nData generation complete....at {data_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a llama model.")
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    regenerate_data(args.data_path)
