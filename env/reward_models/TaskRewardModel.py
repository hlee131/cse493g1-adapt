from env.reward_models.persona_rewards.TaskElementRewardModel import TaskElementRewardModel

class TaskRewardModel(TaskElementRewardModel):
    def __init__(self, interaction_utils):
        super().__init__(interaction_utils)
        task_goal = interaction_utils.task_goal

        penalty = 0
        max_penalty = 0
        messages = []

        if task_goal == 'Make cereal and coffee for breakfast':
            subgoal_success = self.cereal_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Cereal and milk should be mixed.")

            subgoal_success = self.coffee_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Coffee beans should be used to make coffee.")

        elif task_goal == 'Make tea and eggs for breakfast':
            subgoal_success = self.tea_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Tea bags and water should be mixed to make tea.")

            subgoal_success = self.egg_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Eggs should be cooked.")

        elif task_goal == 'Make toast and coffee for breakfast':
            subgoal_success = self.bread_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Bread should be used to create toast.")

            subgoal_success = self.coffee_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Coffee beans should be used to make coffee.")

        elif task_goal == 'Make toast and eggs for breakfast':
            subgoal_success = self.bread_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Bread should be used to create toast.")

            subgoal_success = self.egg_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Egg should be cooked.")

        elif task_goal == 'Make yoghurt parfait for breakfast':
            subgoal_success = self.yoghurt_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Yoghurt should be used.")
        
        elif task_goal == 'Prepare cereal for breakfast':
            subgoal_success = self.cereal_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Cereal and milk should be mixed.")

        elif task_goal == 'Prepare eggs for breakfast':
            subgoal_success = self.egg_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Egg should be cooked.")

        elif task_goal == 'Prepare omelette for breakfast':
            subgoal_success = self.egg_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Egg should be cooked to make omelette.")

        elif task_goal == 'Prepare scrambled eggs for breakfast':
            subgoal_success = self.egg_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Egg should be cooked to make scrambled eggs.")
             
        elif task_goal =='Make pancakes for breakfast':
            subgoal_success = self.pancake_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Pancake mix or ingredients should be cooked to make a pancake.")
            
        elif task_goal =='Make a french toast for breakfast':
            subgoal_success = self.french_toast_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Pancake mix or ingredients should be cooked to make a pancake.")
            
        elif task_goal in ['Make coffee and oatmeal for breakfast', 'Prepare coffee and oatmeal for breakfast']:
            subgoal_success = self.coffee_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Coffee beans should be used to make coffee.")
            subgoal_success = self.oatmeal_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Oatmeal should be used.")
            
        elif task_goal == 'Prepare tea and toast for breakfast':
            subgoal_success = self.tea_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Tea bags and water should be mixed to make tea.")
            
            subgoal_success = self.bread_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Bread should be used to create toast.")

        elif task_goal == 'Prepare tea and cheese toast for breakfast':
            subgoal_success = self.tea_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Tea bags and water should be mixed to make tea.")
            
            subgoal_success = self.bread_created()
            max_penalty += 1
            if not subgoal_success:
                penalty += 1
                messages.append("Cheese should be used on toast.")

        else:
            raise ValueError(f"Task goal {task_goal} not recognized. Must be one of: \n'Make cereal and coffee for breakfast', \n'Make tea and eggs for breakfast', \n'Make toast and coffee for breakfast', \n'Make toast and eggs for breakfast', \n'Make yoghurt parfait for breakfast', \n'Prepare cereal for breakfast', \n'Prepare eggs for breakfast', \n'Prepare omelette for breakfast' \n'Make pancakes for breakfast', \n'Make a french toast for breakfast', \n'Make coffee and oatmeal for breakfast', \n'Prepare tea and toast for breakfast'")

        self.penalty = penalty
        self.max_penalty = max_penalty
        self.messages = messages
        self.success = 1 - (penalty/max_penalty)
