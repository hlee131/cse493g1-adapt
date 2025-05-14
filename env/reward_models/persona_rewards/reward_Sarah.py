import env.reward_models.persona_rewards.object_list_examples as obj_list
from env.reward_models.persona_rewards.TaskElementRewardModel import TaskElementRewardModel
import env.reward_models.persona_rewards.task_lists as task_list

class RewardModel():
    def __init__(self, interaction_utils):
        task_goal = interaction_utils.task_goal
        is_object_left_at_location = interaction_utils.is_object_left_at_location
        is_action_performed = interaction_utils.is_action_performed
        is_available = interaction_utils.is_available
        is_used_if_available = interaction_utils.is_used_if_available
        is_full = interaction_utils.is_full
        are_objects_mixed = interaction_utils.are_objects_mixed
        is_used = interaction_utils.is_used
        is_used_before = interaction_utils.is_used_before
        is_served_before = interaction_utils.is_served_before
        is_cooked = interaction_utils.is_cooked
        are_objects_mixed_in_order = interaction_utils.are_objects_mixed_in_order
        is_created = interaction_utils.is_created
        is_added_while_preparing = interaction_utils.is_added_while_preparing
        all_final_food_location = interaction_utils.all_final_food_location
        is_added_to_ingredient_or_while_preparing = interaction_utils.is_added_to_ingredient_or_while_preparing
        task_check = TaskElementRewardModel(interaction_utils)

        self.preference_functions_and_messages = []

        ## Eggs must be loaded with toppings such as vegetables ##
        def eggs_with_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_vegetables, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_with_vegetables, "Eggs must be loaded with toppings such as vegetables"))
        
        ## Eggs must be loaded with toppings such as cheese ##
        def eggs_with_cheese(list_of_eggs, list_of_cheese, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_cheese, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_with_cheese, "Eggs must be loaded with toppings such as cheese"))
        
        ## Pair a toast with spread ##
        def toasted_bread_with_spreads(list_of_bread, list_of_spreads, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_created('toast'):
                if not are_objects_mixed(list_of_spreads, list_of_bread):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((toasted_bread_with_spreads, "Pair a toast with spread"))
        
        ## Add fruit as topping with sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def fruits_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_fruits):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fruits_on_sweet_breakfasts, "Add nuts as a topping to sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Sprinkle cinnamon on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def cinnamon_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cinnamon, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_cinnamon):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((cinnamon_on_sweet_breakfasts, "Sprinkle cinnamon on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Drizzle honey or maple syrup on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def honey_or_maple_syrup_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_honey, list_of_maple_syrups, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used(list_of_honey) or is_used(list_of_maple_syrups):
                return 'satisfied'
            if not is_available(list_of_honey) and not is_available(list_of_maple_syrups):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((honey_or_maple_syrup_on_sweet_breakfasts, "Drizzle honey or maple syrup on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Add fresh herbs to eggs ##
        def fresh_herbs(list_of_eggs, list_of_fresh_herbs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_used(list_of_fresh_herbs):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fresh_herbs, "Add fresh herbs to eggs"))
        
        ## Always use cheese and vegetables together, if making eggs ##
        def mix_and_match_cheese_and_vegetables(list_of_cheese, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_cheese, list_of_vegetables):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((mix_and_match_cheese_and_vegetables, "Always use cheese and vegetables together, if making eggs"))
        
        ## When preparing cereal, add cereal first then milk, in that order ##
        def add_cereal_then_milk(list_of_cereals, list_of_milk, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if not are_objects_mixed_in_order(list_of_cereals, list_of_milk):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((add_cereal_then_milk, "When preparing cereal, add cereal first then milk, in that order"))

        ## Prefers to dine on the patio ##
        def dine_on_patio(list_of_patio_tables, **kwargs):
            if all_final_food_location(list_of_patio_tables):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((dine_on_patio, "Prefers to dine on the patio"))
        
        penalty = 0
        max_penalty = 0
        messages = []
        messages_task = []

        for preference_functions_and_message in self.preference_functions_and_messages:
            preference_function, message = preference_functions_and_message
            preference_response = preference_function(**obj_list.__dict__)
            if preference_response in ['violated', 'not preferred', 'unsatisfied']:
                penalty += 1
                max_penalty += 1
                messages.append(f"FAILED: {message}")
            elif preference_response in ['satisfied', 'preferred']:
                max_penalty += 1
                messages.append(f"SUCCEEDED: {message}")
            elif preference_response in ['inapplicable','neutral']:
                messages.append(f"NOT RELEVANT: {message}")
            else:
                assert isinstance(preference_response, str),  f"Invalid response: {preference_response}, {message}"
                messages_task.append(f"FAILED: {preference_response}")
                penalty += 1
                max_penalty += 1
                messages.append(f"FAILED: {message}")
                 
        self.penalty = penalty
        self.max_penalty = max_penalty
        # self.messages = list(set(messages_task)) + messages
        self.messages = messages