import os
import shutil
import time
import json
import argparse
import numpy as np
import dill as pickle
import multiprocessing as mp
from copy import deepcopy
from env.scene import SceneGenerator
import evaluate
from src.LLMAgent import LLMAgent, LLMAgent_Planner, LLMAgent_Persona, summarize_standalone, prompt_from_rollout
from src.run_task import run_task
from src.Environment import Environment
from src.utils import *
from env.scene import scene_from_key
from src.run_summary import summarize


def run_eval_interaction(data_filepath, persona_id, env_key, goal, parameters):
    os.makedirs(os.path.dirname(data_filepath), exist_ok=True)
    summary_filepath = data_filepath.replace("/interaction_", "/summary_")
    os.makedirs(os.path.dirname(summary_filepath), exist_ok=True)
    scene = scene_from_key(env_key)
    if os.path.exists(data_filepath):
        data = json.load(open(data_filepath))
        if data['finished']:
            print(f"Run found at {data_filepath} is complete. Skipping...")
            return
        print(f"Run found at {data_filepath} is incomplete. Continuing...")
    run_task(scene, goal, persona_id, None, data_filepath, parameters, {'initial_scene_key':env_key})
        

def rollout_eval_interactions(run_config, ctx, **args):
    processes = []
    for persona_id in run_config["personas"]:
        for eval_idx, goal in enumerate(run_config["goals"]):
            env_key = run_config["environment_keys"][persona_id][eval_idx][0]
            interaction_filepath = os.path.join(args['logs_dir'], f"{persona_id}_eval{eval_idx}_{goal.replace(' ','_')}.json")
            if os.environ.get('USE_MULTIPROCESSING',None) == '1':
                p = ctx.Process(target=run_eval_interaction, args=(interaction_filepath, persona_id, env_key, goal, parameters))
                processes.append(p)
                p.start()
            else:
                run_eval_interaction(interaction_filepath, persona_id, env_key, goal, parameters)
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a rollout.')
    parser.add_argument('--generalization_category', type=str, choices=['seen_persona', 'unseen_persona'], required=True)
    parser.add_argument('--crossvalidation_split', type=int, choices=[0,1,2,3], required=True)
    parser.add_argument('--logs_dir', type=str, help='', default='logs')

    parser.add_argument('--model_name_planner', type=str, required=True)
    parser.add_argument('--model_name_base', type=str, required=True)

    parser.add_argument('--progress_summ', action='store_true')
    parser.add_argument('--summarize_todos', action='store_true')
    parser.add_argument('--interleave_thought', action='store_true')
    parser.add_argument('--no_history_planner', action='store_true')
    parser.add_argument('--unconstrained_planner', action='store_true')
    parser.add_argument('--use_privileged_prior', action='store_true')
    parser.add_argument('--no_ask_option', action='store_true')
    parser.add_argument('--user_info_with_summary', action='store_true')
    parser.add_argument("--force_question_every_n", type=int)
    
    args = parser.parse_args()

    ## Evals should always use zero temperature
    args.temperature_planner = 0.0
    args.temperature_persona = 0.0

    run_config = json.load(open(f'cfg/split{args.crossvalidation_split}_run_config_eval_{args.generalization_category}.json'))
    args.logs_dir = os.path.join(args.logs_dir, args.generalization_category)

    ctx = mp.get_context('spawn')

    scene_generator = SceneGenerator(split='eval')
    parameters = deepcopy(run_config)
    if 'personas' in parameters:
        del parameters['personas']
    if 'goals' in parameters:
        del parameters['goals']
    if 'goal_combinations' in parameters:
        del parameters['goal_combinations']
    if 'goal_sets' in parameters:
        del parameters['goal_sets']
    if 'skip' in parameters:
        del parameters['skip']
    os.makedirs(args.logs_dir, exist_ok=True)
    parameters.update(args.__dict__)

    rollout_eval_interactions(run_config, ctx, **parameters)
