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
 
        ## Vegan diet using egg substitutes ##
        def vegan_egg_substitutes(list_of_eggs, list_of_vegan_egg_substitutes, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_vegan_egg_substitutes):
                if not is_used(list_of_vegan_egg_substitutes):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((vegan_egg_substitutes, "Vegan diet using egg substitutes"))
 
        ## Eggs should be prepared with a side of vegetables ##
        def omelette_with_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not is_available(list_of_vegetables):
                return 'inapplicable'
            if not is_used(list_of_vegetables):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((omelette_with_vegetables, "Eggs should be prepared with a side of vegetables"))
        
        ## Avocados are a favorite as a side, and especially necessary on toast ##
        def toast_with_avocado(list_of_toast, list_of_avocados, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_avocados):
                if not is_used(list_of_avocados):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((toast_with_avocado, "Avocados are a favorite as a side, and especially necessary on toast"))
        
        ## Beverages like coffee and tea are preferred is preferred black without any added sugars or creamers ##
        def black_beverages(list_of_coffee_and_tea, list_of_sugars, list_of_creamers, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if are_objects_mixed(list_of_sugars, list_of_coffee_and_tea):
                return 'violated'
            if are_objects_mixed(list_of_creamers, list_of_coffee_and_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((black_beverages, "Beverages like coffee and tea are preferred is preferred black without any added sugars or creamers"))
        
        ## To maintain a vegan diet, use vegan alternatives to milk ##
        def vegan_alternatives(list_of_milks, list_of_plant_based_milks, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_available(list_of_plant_based_milks):
                if not is_used(list_of_plant_based_milks):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((vegan_alternatives, "To maintain a vegan diet, use vegan alternatives to milk"))
                
        ## To maintain a vegan diet, use vegan alternatives to cheese when cooking eggs ##
        def vegan_alternatives(list_of_cheeses, list_of_vegan_cheese, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_vegan_cheese):
                if not is_used(list_of_vegan_cheese):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((vegan_alternatives, "To maintain a vegan diet, use vegan alternatives to cheese when cooking eggs"))
                
        ## To maintain a vegan diet, use vegan alternatives to yoghurts ##
        def vegan_alternatives(list_of_yoghurts, list_of_plant_based_yoghurt, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_plant_based_yoghurt):
                if not is_used(list_of_plant_based_yoghurt):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((vegan_alternatives, "To maintain a vegan diet, use vegan alternatives to yoghurts"))
        
        ## A side of berries is a nice complement for any breakfast meal, no matter whether it is savory or sweet ##
        def berries_as_side(list_of_berries, **kwargs):
            if is_available(list_of_berries):
                if is_used(list_of_berries):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((berries_as_side, "A side of berries is a nice complement for any breakfast meal, no matter whether it is savory or sweet"))
        
        ## A side of nuts is a nice complement for any sweet breakfast ##
        def nuts_as_side(list_of_nuts, list_of_sweet_breakfasts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_available(list_of_nuts):
                if is_used(list_of_nuts):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((nuts_as_side, "A side of nuts is a nice complement for any sweet breakfast"))
        
        ## Serve food directly at the standing desk to start the work day right away ##
        def food_at_desk(list_of_standing_desk, **kwargs):
            if is_used(list_of_standing_desk):
                if not all_final_food_location(list_of_standing_desk):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((food_at_desk, "Serve food directly at the standing desk to start the work day right away"))
        
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
        