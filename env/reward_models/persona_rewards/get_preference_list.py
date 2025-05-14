import importlib
import sys
sys.path.append("/srv/rail-lab/flash5/mpatel377/repos/meta/PolicyPersonalization")
sys.path.append("/opt/hpcaas/.mounts/fs-03ee9f8c6dddfba21/jtruong/maithili")

from env.reward_models.interaction_utils import InteractionUtils

int_utils_empty = InteractionUtils(None)

def get_preference_list(persona_id):
    reward_model_lib = importlib.import_module(f"env.reward_models.persona_rewards.reward_{persona_id}")
    reward_model = reward_model_lib.RewardModel(int_utils_empty)
    open(f"env/reward_models/persona_rewards/reward_list_{persona_id}.txt", "w").writelines([f"{p[1]}\n" for p in reward_model.preference_functions_and_messages])
    return [p[1] for p in reward_model.preference_functions_and_messages]


if __name__ == "__main__":
    for persona in [
        "Jamal",
        "Leila",
        "Carlos",
        "Juan",
        "Lisa",
        "Maya",
        "Nalini",
        "Ramesh",
        "Samantha",
        "Ethan",
        "Mark",
        "Rachel",
        "Miranda",
        "Sarah",
        "Ron",
        "Maria"
    ]:
        get_preference_list(persona)