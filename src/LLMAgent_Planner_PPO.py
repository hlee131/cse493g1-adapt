import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

from src.LLMAgent_Planner_PPO import LLMAgent_Planner_PPO
from src.Environment import Environment
from env.scene import SceneGenerator
from src.utils import _tokenizer as global_tokenizer

# --- Configuration ---
ppo_config_params = {
    "model_name": "path/to/your/sft_tuned_model",  # Your SFT model
    "learning_rate": 1.41e-5,
    "batch_size": 64,
    "mini_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "ppo_epochs": 4,
    "log_with": "wandb",
}
config = PPOConfig(**ppo_config_params)

# --- Model Loading ---
# Load the policy model (your SFT-tuned model)
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True  # Optional: for memory efficiency
)

# Load reference model (same as policy, but frozen)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Initialize PPO Trainer with your models
ppo_trainer = PPOTrainer(
    config=config, 
    model=policy_model, 
    ref_model=ref_model, 
    tokenizer=tokenizer
)

# Initialize PPO Planner Agent
ppo_planner_agent = LLMAgent_Planner_PPO(
    persona_id="PPO_Agent",
    policy_model=ppo_trainer.model,  # Use the model from PPOTrainer
    critic_model=ppo_trainer.critic,  # Use the critic from PPOTrainer
    tokenizer_instance=tokenizer,
    temperature_planner=0.7,
    no_ask_option=False,
    user_info_with_summary=False
)

# --- Training Loop ---
max_ppo_steps = 1000
max_episode_length = 50

for ppo_step in tqdm(range(max_ppo_steps)):
    # Collect batch of experiences
    queries = []
    responses = []
    rewards = []
    values = []
    
    for episode in range(config.batch_size):
        # Initialize new episode
        scene_data = SceneGenerator(split="train")()
        env = Environment(scene_data)
        rollout_past = []
        
        # Get task from your task distribution
        current_task = "make breakfast"  # Replace with your task sampling logic
        
        # Reset agent state
        ppo_planner_agent.reset()
        
        episode_queries = []
        episode_responses = []
        episode_rewards = []
        episode_values = []
        
        for step in range(max_episode_length):
            # Collect one step
            query_tensor, response_tokens, value, interaction_data = ppo_planner_agent.collect_rollout_step(
                env, rollout_past, current_task
            )
            
            # Calculate reward (implement your reward function)
            step_reward = calculate_reward(interaction_data, env, rollout_past)
            
            # Store step data
            episode_queries.append(query_tensor)
            episode_responses.append(response_tokens)
            episode_rewards.append(torch.tensor(step_reward, device=ppo_trainer.accelerator.device))
            episode_values.append(value)
            
            # Update rollout history
            rollout_past.append(interaction_data)
            
            # Check for episode termination
            if interaction_data["action_enum"] == "done" or step == max_episode_length - 1:
                break
        
        # Add episode data to batch
        queries.extend(episode_queries)
        responses.extend(episode_responses)
        rewards.extend(episode_rewards)
        values.extend(episode_values)
    
    # Perform PPO update
    stats = ppo_trainer.step(queries, responses, rewards)
    ppo_trainer.log_stats(stats, batch={"queries": queries, "responses": responses}, rewards=rewards)

# Save the trained model
ppo_trainer.save_pretrained("path/to/your/ppo_trained_model")

def calculate_reward(interaction_data, env, rollout_past):
    """
    Implement your reward function here.
    This should return a scalar reward for the current step.
    """
    reward = 0.0
    
    # Basic success reward
    if interaction_data["success"]:
        reward += 1.0
    else:
        reward -= 0.5
    
    # Task completion bonus
    if interaction_data["action_enum"] == "done":
        reward += 10.0
    
    # Add your task-specific reward logic here
    # You can use process_trace for terminal rewards if needed
    
    return reward