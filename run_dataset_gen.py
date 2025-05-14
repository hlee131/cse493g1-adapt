import os
import shutil
import random
import time
import json
import argparse
import multiprocessing as mp
from copy import deepcopy
from env.scene import SceneGenerator
from env.scene import key_from_scene
from src.run_task_with_ref_model import run_task_with_ref
from src.utils import *
from src.run_summary import summarize



def run_second_interactions(run_config,
                            parameters,
                            ctx):

    parameters['out_dir'] = os.path.join(parameters['out_dir'], 'summary_data')
    os.makedirs(parameters['out_dir'], exist_ok=True)
    
    processes = []
    for idx_persona, persona_id in enumerate(run_config["personas"]):
        for idx_goal, goal_for_summary in enumerate(run_config["goals"]):
            if parameters['fixed_goal_seq']:
                goal = json.load('goal2.json')[persona_id+goal_for_summary]
            else:
                goal = random.choice([g for g in run_config["goals_second"] if g != goal_for_summary])
            print("\n\n======================================================================================================================================================\n")
            print(f"Rolling out second interaction for persona {persona_id} towards goal {goal}")
            print("\n======================================================================================================================================================\n\n")
            summary_file = f"{persona_id}_{goal_for_summary.replace(' ','_')}_0.json"
            summary_filepath = os.path.join(parameters['out_dir'].replace('summary_data','summaries'),summary_file)
            print(f"Summary file: {summary_file}")
            if not os.path.exists(summary_file):
                summarize(summary_filepath.replace("/summaries/","/data/"), summary_filepath, parameters['model_name_base'], temperature=0.0, return_n=1)
            summary_data = json.load(open(summary_filepath))
            for g in run_config["goals_second"]:
                data_filepath = os.path.join(parameters['out_dir'], f"{summary_file.split('.')[0]}_{0:02d}_{g.replace(' ','_')}.txt").replace('.txt',f'_0.json')
                if os.path.exists(data_filepath):
                    goal = g
                    break
            summaries = summary_data['summary']
            for idx_summary, summary in enumerate(summaries):
                if idx_summary >= parameters['num_summaries']: continue
                log_filepath = os.path.join(parameters['out_dir'], f"{summary_file.split('.')[0]}_{idx_summary:02d}_{goal.replace(' ','_')}.txt")
                print("---------------------------------------------------------------------------------------------------------------------------")
                print(f"  - Progess:")
                print(f"          - Persona: {idx_persona+1}/{len(run_config['personas'])}")
                print(f"          - Goal: {idx_goal+1}/{len(run_config['goals'])}")
                print(f"          - Summary: {idx_summary+1}/{len(summaries)}")
                print(f"  - Summary: {summary}")
                print(f"  - Output file: {log_filepath.replace('.txt','...')}")
                print("---------------------------------------------------------------------------------------------------------------------------")

                for rollout_num in range(parameters['num_planner_rollouts']):
                    data_filepath = log_filepath.replace('.txt',f'_{rollout_num}.json')
                    scene = SceneGenerator(split="train")()
                    if os.path.exists(data_filepath): 
                        if parameters['overwrite_data']:
                            os.remove(data_filepath)
                        elif json.load(open(data_filepath))['finished']:
                            print(f"Run found at {data_filepath}. Moving on...")
                            continue
                        else:
                            print(f"Run found at {data_filepath} is incomplete. Continuing...")
                            # os.remove(data_filepath)
                
                    if os.environ.get('USE_MULTIPROCESSING',None) == '1':
                        p = ctx.Process(target=run_task_with_ref, args=(scene, 
                                            goal, 
                                            persona_id, 
                                            summary, 
                                            data_filepath, 
                                            parameters, 
                                            {'prior_summary_prompt':summary_data['prompt'], 'prior_summary_tasks':summary_data['prior_tasks'], 'initial_scene_key':key_from_scene(scene), 'summary_source':(summary_file, idx_summary)}, 
                                            data_filepath.replace(parameters['out_dir'],parameters['ref_dir']) if parameters['ref_dir'] is not None else None, 
                                            parameters['unconstrained_student']))
                        processes.append(p)
                        p.start()
                    else:
                        run_task_with_ref(scene, 
                                            goal, 
                                            persona_id, 
                                            summary, 
                                            data_filepath, 
                                            parameters, 
                                            {'prior_summary_prompt':summary_data['prompt'], 'prior_summary_tasks':summary_data['prior_tasks'], 'initial_scene_key':key_from_scene(scene), 'summary_source':(summary_file, idx_summary)}, 
                                            data_filepath.replace(parameters['out_dir'],parameters['ref_dir']) if parameters['ref_dir'] is not None else None, 
                                            parameters['unconstrained_student'])
    for p in processes:
        p.join()       



def run_first_interaction(run_config,
                          parameters,
                          ctx):
    parameters['out_dir'] = os.path.join(parameters['out_dir'], 'data')
    os.makedirs(parameters['out_dir'].replace('data','summaries'), exist_ok=True)
    os.makedirs(parameters['out_dir'], exist_ok=True)
    processes = []
    for idx_persona, persona_id in enumerate(run_config["personas"]):
        for idx_goal, goal in enumerate(run_config["goals"]):
            print("\n\n======================================================================================================================================================\n")
            print(f"Rolling out first interaction for persona {persona_id} towards goal {goal}")
            print("\n======================================================================================================================================================\n\n")
            log_filepath = os.path.join(parameters['out_dir'], f"{persona_id}_{goal.replace(' ','_')}.txt")
            print("---------------------------------------------------------------------------------------------------------------------------")
            print(f"  - Progess:")
            print(f"          - Persona: {idx_persona+1}/{len(run_config['personas'])}")
            print(f"          - Goal: {idx_goal+1}/{len(run_config['goals'])}")
            print(f"  - Output file: {log_filepath.replace('.txt','...')}")
            print("---------------------------------------------------------------------------------------------------------------------------")

            for rollout_num in range(parameters['num_planner_rollouts']):
                data_filepath = log_filepath.replace('.txt',f'_{rollout_num}.json')
                scene = SceneGenerator(split="train")()
                if os.path.exists(data_filepath): 
                    if parameters['overwrite_data']:
                        os.remove(data_filepath)
                    elif json.load(open(data_filepath))['finished']:
                        print(f"Run found at {data_filepath}. Moving on...")
                        continue
                    else:
                        print(f"Run found at {data_filepath} is incomplete. Continuing...")
                        # os.remove(data_filepath)
            
                if os.environ.get('USE_MULTIPROCESSING',None) == '1':
                    p = ctx.Process(target=run_task_with_ref, args=(scene, 
                                    goal, 
                                    persona_id, 
                                    None, 
                                    data_filepath, 
                                    parameters, 
                                    {'summary_history':{'initial':{'task':None,'persona':None},'interactions':[]}, 'initial_scene_key':key_from_scene(scene)}, 
                                    data_filepath.replace(parameters['out_dir'],parameters['ref_dir']) if parameters['ref_dir'] is not None else None, 
                                    parameters['unconstrained_student']))
                    processes.append(p)
                    p.start()
                else:
                    run_task_with_ref(scene, 
                                    goal, 
                                    persona_id, 
                                    None, 
                                    data_filepath, 
                                    parameters, 
                                    {'summary_history':{'initial':{'task':None,'persona':None},'interactions':[]}, 'initial_scene_key':key_from_scene(scene)}, 
                                    data_filepath.replace(parameters['out_dir'],parameters['ref_dir']) if parameters['ref_dir'] is not None else None, 
                                    parameters['unconstrained_student'])
    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a rollout.')
    parser.add_argument('--run_config', type=str, required=True)
    parser.add_argument('--ref_dir', type=str)
    parser.add_argument('--out_dir', type=str, help='', default='data')

    parser.add_argument('--model_name_planner', type=str, required=True)
    parser.add_argument('--model_name_base', type=str, required=True)

    # Overwritable configs
    parser.add_argument('--temperature_planner', type=float, default='0.0')
    parser.add_argument('--temperature_persona', type=float, default='0.0')
    parser.add_argument('--use_privileged_prior', action='store_true')
    parser.add_argument('--progress_summ', action='store_true')
    parser.add_argument('--summarize_todos', action='store_true')
    parser.add_argument('--overwrite_data', action='store_true')
    parser.add_argument('--interleave_thought', action='store_true')
    parser.add_argument('--action_only_teacher', action='store_true')
    parser.add_argument('--unconstrained_student', action='store_true')
    parser.add_argument('--fixed_goal_seq', action='store_true')
    parser.add_argument('--user_info_with_summary', action='store_true')

    args = parser.parse_args()
    args.num_summaries = 1
    args.num_planner_rollouts = 1

    run_config = json.load(open(args.run_config))
    
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
    
    ctx = None
    if os.environ.get('USE_MULTIPROCESSING',None) == '1':
        ctx = mp.get_context('spawn')

    run_config['goals_second'] = run_config['goals'].copy()

    parameters.update(args.__dict__)
    out_dir = parameters['out_dir']
    run_first_interaction(run_config, parameters, ctx)
    parameters['out_dir'] = out_dir
    run_second_interactions(run_config, parameters, ctx)
    