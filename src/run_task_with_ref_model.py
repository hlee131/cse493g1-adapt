import os
import json
from copy import deepcopy

from src.LLMAgent import LLMAgent_Planner, LLMAgent_Persona
from src.Environment import Environment, parse_action
from src.utils import VERBOSE, VERBOSE_RUNLEVEL, NO_UNDO_OPTION, _tokenizer
from src.convert_and_evaluate import get_privileged_preferences_at_step
from src.Trainer import extract_data_for_planner
from src.run_summary import summarize

max_completion_reward = 10
reward_ask = -1

def step_action(env, current_interaction, persona, steps_remaining, rollout_data, failure_streak):
    task = rollout_data["task"]
    action = current_interaction["action"]
    success, msg, action_enum, action_args = env.step(action)
    current_interaction["success"] = success
    current_interaction["observation"] = msg
    current_interaction["action_enum"] = action_enum
    current_interaction["action_args"] = action_args
    if success:
        steps_remaining -= 1
    if VERBOSE_RUNLEVEL:
        print(f"--------------------------------")
        print(f"Thought: {current_interaction['thought']}\nAction: {action}")
        print(f"Observation: {msg}\n", end="")
    
    reward = None
    rollout_data["episode_length"] += 1
    rollout_data["sim_steps"] = env.step_num
    if msg == "DONE":
        failure_streak = 0
        task_done = True
        try:
            results_dict = persona(env, rollout_data, current_interaction, ask_or_correct='confirm_done', task=task)
            rollout_data["num_pref_violated"] = results_dict["penalty"]
            rollout_data["num_pref_satisfied"] = results_dict["max_penalty"] - results_dict["penalty"]
            user_response_subjective = results_dict["messages"]
            response_objective = results_dict["reward_fraction"]
        except Exception as e:
            user_response_subjective, response_objective = "NOT evaluated", 0
        current_interaction["user_feedback"] = user_response_subjective
        if VERBOSE_RUNLEVEL:
            print(f"User: ({response_objective}) {user_response_subjective}\n")
        reward = response_objective * task_done
        rollout_data['finished'] = True
        rollout_data["goal_completion_fraction"] = response_objective
        rollout_data["goal_completion_reward"] = reward
    elif not success:
        pass
    elif action_enum in ["ask"]:
        failure_streak = 0
        user_response_subjective, response_objective = persona(
            env, rollout_data, current_interaction, ask_or_correct="ask", task=task
        )
        current_interaction["user_feedback"] = user_response_subjective
        if VERBOSE_RUNLEVEL:
            print(f"User: ({response_objective}) {user_response_subjective}\n")
        reward = reward_ask
        rollout_data["num_questions"] += 1
    elif action_enum in [
        "move",
        "move_from",
        "mix",
        "cook",
        "pour",
        "pour_into",
        "peel",
        "place",
        "serve",
        "chop",
        "chop_obj",
        "freeform",
        "freeform_contents",
        "heat",
        "turn_on",
        "turn_off",
    ]:
        failure_streak = 0
    elif action_enum in ['undo']:
        assert not NO_UNDO_OPTION, f"The model wasn't supposed to be undoing actions!!"
    else:
        failure_streak += 1
        reward = 0
        if action_enum not in [
            "find",
            "search",
            "done",
            "open",
            "close",
            "look_for",
            "search_to_find",
        ]: 
            print(f"WARNING: You forgot to handle action {action_enum} in invoking the persona agent!!!")
    
    current_interaction["reward"] = reward

    rollout_data["rollout"].append(current_interaction)
    if reward is not None:
        rollout_data["total_reward"] += reward

    return env, rollout_data


def run_task_with_ref(
    scene,
    task,
    persona_id,
    prior_user_info,
    data_filepath,
    parameters,
    additional_data={},
    ref_filepath=None,
    unconstrained_student=False,
):
    env = Environment(scene)
    max_steps_executed = parameters['max_steps_executed']
    max_steps_predicted = parameters['max_steps_predicted']
    action_only_planner = (not parameters['interleave_thought']) if 'interleave_thought' in parameters else False
    action_only_teacher = parameters['action_only_teacher'] if 'action_only_teacher' in parameters else False
    remove_failures_in_prompt = parameters['remove_failures_in_prompt'] if 'remove_failures_in_prompt' in parameters and parameters['remove_failures_in_prompt'] is not None else False
    persona = LLMAgent_Persona(persona_id, **parameters)
    planner = LLMAgent_Planner(persona_id, no_ask_option=False, **parameters)
    priv_parameters = deepcopy(parameters)
    priv_parameters['model_name_planner'] = parameters['model_name_base']
    privileged_planner = LLMAgent_Planner(persona_id, no_ask_option= True, **priv_parameters)
    planner.add_user_info(prior_user_info)
    privileged_planner.add_user_info(persona.privileged_preferences)
    steps_remaining = max_steps_executed
    max_steps_w_failure = max_steps_predicted
    rollout_data = {
        "task": task,
        "persona_id": persona.agent_name,
        "finished": False,
        "summary_history": None,
        "privileged_goal": None,
        "prior_persona_knowledge": None,
        "total_reward": 0,
        "num_questions": 0,
        "num_corrections": 0,
        "goal_completion_reward": 0,
        "goal_completion_fraction": 0,
        "sim_steps": 0,
        "episode_length": 0,
        "rollout": [],
        "summary": None,
        "initial_scene": json.dumps(env.full_scene),
        "actions_performed": None,
        "entities_created": None,
        "objects_used": None,
        "mixtures": None,
        "transformations": None,
        "final_object_locations": None,
    }
    rollout_data.update(additional_data)
    persona.task = task
    rollout_data["prior_persona_knowledge"] = planner.user_info
    reference_rollout = None
    if ref_filepath is not None and os.path.exists(ref_filepath):
        reference_rollout = json.load(open(ref_filepath))
    if os.path.exists(data_filepath):
        rollout_data = json.load(open(data_filepath))
        env = Environment(json.loads(rollout_data["initial_scene"]))
        if reference_rollout is not None:
            assert reference_rollout['initial_scene'] == rollout_data['initial_scene'], "Initial scene mismatch between reference and partial rollout"
    output_filepath = data_filepath.split("summary_data/")[0].split("data/")[0]+"data_with_labels_and_probs.json"
    failure_streak = 0
    privileged_info_at_step = None
    for step in range(max_steps_w_failure):
        data_out = {k:None for k in ['probabilities','as','at','ques','ans','xs','xt','xs_q','xt_q','grammar']}
        if len(rollout_data["rollout"]) > step:
            if rollout_data["rollout"][step]["success"]:
                action = rollout_data["rollout"][step]["action"]
                env.step(action)
                continue
        if VERBOSE_RUNLEVEL:
            print(f"******** Step {step} ********")

        progress_summ = env.summarize_progress()
        if reference_rollout is not None and len(reference_rollout['rollout']) > step:
            prompt = reference_rollout['rollout'][step]['planner_prompt']
            initial_thought = reference_rollout['rollout'][step]['thought']
            initial_action = reference_rollout['rollout'][step]['action']
        else:
            initial_action, initial_thought, grammar_verify, prompt, prob_response_as = planner(env, 
                                                rollout_data["rollout"], 
                                                task, 
                                                progress_summ=progress_summ, 
                                                action_only=action_only_planner,
                                                no_generate=False,
                                                unconstrained_planner=unconstrained_student)
        action = initial_action

        data_out['xs'] = prompt
        data_out['as'] = f"Thought: {initial_thought}\nAction: {initial_action}"
        data_out['grammar'] = grammar_verify

        ## list of action,action_args
        actions_maybe = list(parse_action(action))
        reflection, question, change = None, None, False
        action_priv, thought_priv = None, None
        action_enums = [a[0] for a in actions_maybe]
        privileged_info_at_step = get_privileged_preferences_at_step(rollout_data)
        action_priv, thought_priv, grammar, planner_prompt, prob_response_at = privileged_planner(env,
                                                    rollout_data["rollout"],
                                                    task,
                                                    progress_summ=progress_summ,
                                                    privileged_info_at_step=privileged_info_at_step,
                                                    action_only=action_only_teacher,
                                                    no_generate=False)
        action_privs_maybe = list(parse_action(action_priv))

        data_out['xt'] = planner_prompt
        data_out['at'] = f"Thought: {thought_priv}\nAction: {action_priv}"
        if not any([a1 == a2 for a2 in action_privs_maybe for a1 in actions_maybe]):
            if VERBOSE_RUNLEVEL:
                print(f"\n--------------------------------")
                print(f"Privileged thought: {thought_priv}")
                print(f"Privileged action: {action_priv}")
                print(f"Actual thought: {initial_thought}")
                print(f"Actual action: {initial_action}")
            question = planner.reflection(initial_action, 
                                            initial_thought, 
                                            action_priv, 
                                            thought_priv, 
                                            env, 
                                            rollout_data["rollout"], 
                                            task, 
                                            action_only=action_only_planner)
            thought_ask = f"I need to ask the user \"{question}\""
            action_ask = f"Ask \"{question}\""
            if VERBOSE_RUNLEVEL:
                print(f"Question suggestion: {question}")
            interaction_ask = {"step": env.step_num,
                            "success": True,
                            "progress_summary": progress_summ,
                            "privileged_info": privileged_info_at_step,
                            "grammar": grammar,
                            "thought": thought_ask,
                            "action": action_ask,
                            "privileged_thought": thought_priv,
                            "privileged_action": action_priv,
                            "initial_thought": initial_thought,
                            "initial_action": initial_action,
                            "reflection_question": question,
                            "action_enum": None,
                            "action_args": None,
                            "observation": None,
                            "user_feedback": None,
                            "reward": None,
                            "planner_prompt": planner_prompt,
                            }
            data_out['ques'] = f"Thought: {thought_ask}\nAction: {action_ask}"
            user_response_subjective, _ = persona(
                env, rollout_data, interaction_ask, ask_or_correct="ask", task=task
            )
            interaction_ask["user_feedback"] = user_response_subjective
            data_out['ans'] = user_response_subjective
            
            _,_,_,unprivileged_prompt,_ = planner(env,
                                    rollout_data["rollout"]+[interaction_ask],
                                    task,
                                    progress_summ=progress_summ,
                                    no_generate=True,
                                    action_only=action_only_planner,
                                    skip_failed=remove_failures_in_prompt)
            data_out['xs_q'] = _tokenizer.apply_chat_template(
                                    [{"role": p[0], "content": p[1]} for p in unprivileged_prompt],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
            
            _,_,_,privileged_prompt,_ = planner(env,
                            rollout_data["rollout"]+[interaction_ask],
                            task,
                            progress_summ=progress_summ,
                            privileged_info_at_step=privileged_info_at_step,
                            no_generate=True,
                            skip_failed=remove_failures_in_prompt)
            data_out['xt_q'] =  _tokenizer.apply_chat_template(
                                    [{"role": p[0], "content": p[1]} for p in privileged_prompt],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                )
            
            probabilities = {}
            probabilities['p_s_a_s'] = prob_response_as
            probabilities['p_t_a_t'] = prob_response_at
            probabilities_ref, prompts, completions = planner.reflection_action_as_student(env, 
                                                            action_priv, 
                                                            thought_priv, 
                                                            initial_action, 
                                                            initial_thought, 
                                                            rollout_data["rollout"], 
                                                            interaction_ask, 
                                                            task, 
                                                            action_only=action_only_planner,
                                                            progress_summ=progress_summ)
            probabilities.update(probabilities_ref)
            data_out['probabilities'] = probabilities
            
        open(output_filepath,'a').write(json.dumps(data_out)+'\n')
           
        original_interaction = {
                    "step": env.step_num,
                    "progress_summary": progress_summ,
                    "privileged_info": privileged_info_at_step,
                    "grammar": data_out['grammar'],
                    "thought": initial_thought,
                    "action": initial_action,
                    "privileged_thought": thought_priv,
                    "privileged_action": action_priv,
                    "initial_thought": initial_thought,
                    "initial_action": initial_action,
                    "reflection_thought": reflection,
                    "reflection_question": question,
                    "reflection_change": change,
                    "action_enum": None,
                    "action_args": None,
                    "success": None,
                    "observation": None,
                    "user_feedback": None,
                    "reward": None,
                    "planner_prompt": prompt,
                    "prob_response": prob_response_as,
                }
        env, rollout_data = step_action(env, original_interaction, persona, steps_remaining, rollout_data, failure_streak)
        
        if not rollout_data['finished'] and (step >= max_steps_w_failure - 1 or steps_remaining < 0):
            rollout_data["goal_completion_reward"] = 0
            rollout_data["finished"] = True


        # rollout_data["summarize_prompt"] = summarize_prompt(
        #     rollout_data["rollout"], persona.agent_name, planner.user_info
        # )

        if rollout_data["privileged_goal"] is None:
            rollout_data["privileged_goal"] = persona.preferences_list
        rollout_data.update(env.get_full_state())
        json.dump(rollout_data, open(data_filepath, "w"), indent=4)

        if rollout_data["finished"]:
            if VERBOSE:
                print()
                print(f"   - total_reward : {rollout_data['total_reward']}")
                print(f"   - num_questions : {rollout_data['num_questions']}")
                print(f"   - num_corrections : {rollout_data['num_corrections']}")
                print(f"   - goal_completion_reward : {rollout_data['goal_completion_reward']}")
                print(f"   - sim_steps : {rollout_data['sim_steps']}")
                print(f"   - episode_length : {rollout_data['episode_length']}")
                print()
            break

    prompt_txt = extract_data_for_planner(rollout_data, _tokenizer)
    open(os.path.join(data_filepath.split('/')[0], 'online_train_set_SFT_planner.jsonl'), 'a') \
        .write(json.dumps({"text":prompt_txt, "filename":os.path.basename(data_filepath), "reward":rollout_data['total_reward']})+'\n')
    summary_filename = data_filepath.replace("/summary_data/", "/summaries/").replace("/data/", "/summaries/")
    summarize(data_filepath, summary_filename, parameters['model_name_base'], temperature=0.0, return_n=1)
    if 'summary_source' in additional_data:
        open(os.path.join(data_filepath.split('/')[0], 'summary_future_rewards.jsonl'), 'a') \
                .write(json.dumps({'filename':additional_data['summary_source'][0], 
                                'summary_idx':additional_data['summary_source'][1],
                                'reward':rollout_data['total_reward'],
                                'task': task})+'\n')
    
    return deepcopy(planner), deepcopy(persona)