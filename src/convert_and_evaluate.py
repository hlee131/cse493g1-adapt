import sys
sys.path.append('env/reward_models')
sys.path.append('..')
import os
import json
from copy import deepcopy
from src.Environment import Environment
import importlib
from env.reward_models.interaction_utils import InteractionUtils
from env.reward_models.TaskRewardModel import TaskRewardModel

CROSS_PERSONA_IMAGE_PLOT = True
    
lists_created = {}
task_goal = None

personas= [
        "Jamal",
        "Leila",
        "Maria",
        "Carlos",
        "Juan",
        "Lisa",
        "Mark",
        "Maya",
        "Miranda",
        "Nalini",
        "Rachel",
        "Ramesh",
        "Ethan",
        "Samantha",
        "Sarah"
    ]


rootdir = ''
cross_correlations = {}
cross_correlations_old = {}

def process_trace(rollout_data):
    task_goal = rollout_data['task']
    persona_id = rollout_data['persona_id']
    rollout = rollout_data['rollout']
    initial_env = Environment(json.loads(rollout_data['initial_scene']))

    final_env = deepcopy(initial_env)
    for step in rollout:
        if not step['success']: continue
        final_env.step(step['action'])
 
    rollout_data.update(final_env.get_full_state())
    final_object_locations = {}
    for object_id, object_info in final_env.full_scene.items():
        final_object_locations[object_id] = object_info['location']
        for content in final_env._get_contents(object_id, edible_only=True).split(', '):
            if content == '': continue
            final_object_locations[content] = object_id
    
    lists_created['task_goal'] = task_goal
    lists_created['actions_performed'] = rollout_data['actions_performed']
    lists_created['entities_created'] = rollout_data['entities_created']
    lists_created['entities_created_that_remain'] = rollout_data['entities_created_that_remain']
    lists_created['list_of_cooked_objects'] = rollout_data['objects_cooked']
    lists_created['serving_order'] = rollout_data['serving_order']
    lists_created['list_of_used_objects'] = rollout_data['objects_used']
    lists_created['list_of_entities_created'] = rollout_data['entities_created']
    lists_created['available_objects'] = list(initial_env.full_scene.keys())
    lists_created['mixed_object_sets_unique'] = [mixture['sources'] for mixture in rollout_data['mixtures']]
    lists_created['transformations'] = rollout_data['transformations']
    lists_created['final_object_locations'] = final_object_locations
    lists_created['object_contents_only_edible'] = {k: final_env._get_contents(k, edible_only=True).split(', ') for k in final_env.full_scene.keys()}
    lists_created['object_contents_including_inedible'] = {k: final_env._get_contents(k).split(', ') for k in final_env.full_scene.keys()}

    if not os.path.exists(f"env/reward_models/persona_rewards/reward_{persona_id}.py"): 
        print(f"WARNING! No reward model found for {persona_id}.")
        return {'penalty':0, 
                'max_penalty':0, 
                'reward_fraction':0, 
                'messages':'',
                'task_completion_fraction':0,
                'task_violations':'',
                'preferences_violated':[],
                'preferences_satisfied':[],
                'num_corrections': rollout_data['num_corrections'], 
                'num_questions': rollout_data['num_questions'], 
                'sim_steps': rollout_data['sim_steps'], 
                'episode_length': rollout_data['episode_length'], 
                }, final_env.get_full_state()
    interaction_utils = InteractionUtils(lists_created)
    reward_model_lib = importlib.import_module(f"env.reward_models.persona_rewards.reward_{persona_id}")
    reward_model = reward_model_lib.RewardModel(interaction_utils)
    task_comp_model = TaskRewardModel(interaction_utils)

    reward_frac = 1-(reward_model.penalty/(reward_model.max_penalty + 1e-8))
    list_of_violated_preferences = [m.replace('FAILED: ', '') for m in reward_model.messages if m.startswith('FAILED: ')]
    if len(list_of_violated_preferences) > 0:
        feedback = "The robot violated the following preferences:\n"
        feedback += '\n'.join(list_of_violated_preferences)
    else:
        feedback = "The robot did not violate any preferences."
    list_of_satisfied_preferences = [m.replace('SUCCEEDED: ', '') for m in reward_model.messages if m.startswith('SUCCEEDED: ')]
    if len(list_of_satisfied_preferences) > 0:
        feedback += "\n\nThe robot satisfied the following preferences:\n"
        feedback += '\n'.join(list_of_satisfied_preferences)
    else:
        feedback += "\n\nThe robot did not satisfy any preferences."

    if not abs(reward_frac - len(list_of_satisfied_preferences)/(len(list_of_satisfied_preferences)+len(list_of_violated_preferences)))<0.001:
        print(f"ISSUE: reward_frac: {reward_frac} vs {len(list_of_satisfied_preferences)/(len(list_of_satisfied_preferences)+len(list_of_violated_preferences))} len(list_of_satisfied_preferences): {len(list_of_satisfied_preferences)}, len(list_of_violated_preferences): {len(list_of_violated_preferences)}")

    task_comp_frac = 1-(task_comp_model.penalty/(task_comp_model.max_penalty + 1e-8))
    task_violations = '\n'.join(task_comp_model.messages)

    results_dict = {'penalty':reward_model.penalty, 
                    'max_penalty':reward_model.max_penalty, 
                    'reward_fraction':reward_frac, 
                    'messages':feedback,
                    'task_completion_fraction':task_comp_frac,
                    'task_violations':task_violations,
                    'preferences_violated':list_of_violated_preferences,
                    'preferences_satisfied':list_of_satisfied_preferences,
                    'num_corrections': rollout_data['num_corrections'], 
                    'num_questions': rollout_data['num_questions'], 
                    'sim_steps': rollout_data['sim_steps'], 
                    'episode_length': rollout_data['episode_length'], 
                }

    return results_dict, final_env.get_full_state()


def get_privileged_preferences_at_step(rollout_data):
    priv_prefs_msg = ''
    results_reward, _ = process_trace(rollout_data)
    for pref in results_reward['preferences_violated']:
        priv_prefs_msg += pref + ', '
    if len(priv_prefs_msg) > 0:
        priv_prefs_msg = f"Remember, {rollout_data['persona_id']}'s preferences include: " + priv_prefs_msg[:-2]+"."
    return priv_prefs_msg
