import json
from src.LLMAgent import LLMAgent, summarize_standalone_n_samples
from src.convert_and_evaluate import process_trace
from src.utils import *


def summarize(rollout_file, summary_filename, llm_runname, temperature, return_n=1, target_task=None):
    summarizer = LLMAgent(llm_runname)
    summaries = []
    
    interaction_data = json.load(open(rollout_file))
    persona_id = interaction_data['persona_id']
    goals = interaction_data['task']

    if interaction_data["rollout"][-1]["action_enum"] != "done": 
        interaction_data["rollout"].append({"thought":"declare done to end the episode before summarizing", "action":"declare done", "action_enum":"done" ,"success":True, "user_feedback": "NOT evaluated"})

    if interaction_data["rollout"][-1]["action_enum"] == "done" and interaction_data["rollout"][-1]["user_feedback"] == "NOT evaluated":
        interaction_data["rollout"][-1]["user_feedback"] = None
        interaction_data["rollout"][-1]["reward"] = None
        print(f"\nEvaluating list of preferences:")
        results_dict,_ = process_trace(interaction_data)
        goal_reward_frac = results_dict['reward_fraction']
        goal_reward = goal_reward_frac * 50
        interaction_data["rollout"][-1]["user_feedback"] = results_dict['messages']
        interaction_data["rollout"][-1]["reward"] = goal_reward
        interaction_data["goal_completion_reward"] = goal_reward
        interaction_data["goal_completion_fraction"] = goal_reward_frac
        interaction_data["num_pref_violated"] = results_dict["penalty"]
        interaction_data["num_pref_satisfied"] = results_dict["max_penalty"] - results_dict["penalty"]
        reward = sum([r['reward'] for r in interaction_data["rollout"][:-1] if r['reward'] is not None]) + interaction_data["goal_completion_reward"]
        interaction_data["total_reward"] = reward
        json.dump(interaction_data, open(rollout_file, 'w'), indent=4)
    
    ## Remove the comprehensive summary
    interaction_data['rollout'][-1]['user_feedback'] = None
    
    prompt = summarize_prompt(rollout_past=[interaction_data['rollout']], persona_id=persona_id, user_info='')
    summaries_by_task = []
    for idx, summary in enumerate(summarize_standalone_n_samples(summarizer.run_llm, prompt=prompt, return_n=return_n, temperature=temperature)):
        print(f"-------------------- SUMMARY OPTION {idx+1} --------------------\n{summary}\n")
        summaries.append(summary)
        summaries_by_task.append({})
    
        if target_task is not None:
            prompt = [
                (
                    "system",
                    f"You are an expert at task planning to provide personalized assistance to a user. You have gathered the following information about the user from prior interactions with them. You will create a concise list of preferences relevant to your task at hand\n\n",
                ),
                (
                    "user",
                    f"You know the following about {persona_id}:{summary}",
                ),
                (
                    "user",
                    f"Create a concise list of preferences relevant to the task: {target_task}",
                )
            ]
            for llm_output in summarizer.run_llm(prompt_msgs=prompt, max_tokens=1024, return_n=return_n, temperature=temperature, stops=["Action:", "Thought:", "Observation:", "User:"]):
                response, probs = llm_output["generation"], llm_output["mean_prob"]
                response = response.strip()
                summaries_by_task[-1][target_task] = response
                break


    summary_info = {
        'summary': summaries,
        'summary_by_task': summaries_by_task,
        'persona_id': persona_id,
        'prior_tasks': goals,
        'prompt': prompt
    }

    os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
    json.dump(summary_info, open(summary_filename, 'w'), indent=4)

    return summary_info