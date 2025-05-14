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

        ## Use whole-grain bread when making toast ##
        def whole_grain_toast(list_of_toast, list_of_whole_grain_toast, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_whole_grain_toast):
                if not is_used(list_of_whole_grain_toast):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((whole_grain_toast, "Use whole-grain bread when making toast"))
        
        ## Make toast without butter or spreads ##
        def toast_without_butter_or_spreads(list_of_toast, list_of_butter, list_of_spreads, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_used(list_of_butter) or is_used(list_of_spreads):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((toast_without_butter_or_spreads, "Make toast without butter or spreads"))
        
        ## Make coffee black, without milk or sweetner ##
        def strong_and_black_coffee(list_of_coffee, list_of_milk, list_of_sweetner, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if are_objects_mixed(list_of_milk, list_of_coffee) or are_objects_mixed(list_of_sweetner, list_of_coffee):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((strong_and_black_coffee, "Make coffee black, without milk or sweetner"))
        
        ## Sugar-free pancakes, waffles or French toast ##
        def sugar_free_pancakes_or_waffles_or_french_toast(list_of_pancakes, list_of_french_toast, list_of_waffles, list_of_sugarfree_pancakes, list_of_sugarfree_waffles, list_of_full_calorie_sweetners, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used(list_of_french_toast) or is_created('french toast'):
                if not is_added_while_preparing(list_of_full_calorie_sweetners, 'french toast'):
                    return 'satisfied'
            if is_used(list_of_pancakes) or is_created('pancake'):
                if (is_used(list_of_sugarfree_pancakes) or not is_available(list_of_sugarfree_pancakes)) or is_added_while_preparing(list_of_full_calorie_sweetners, 'pancake'):
                    return 'satisfied'
            if is_used(list_of_waffles) or is_created('waffle'):
                if (is_used(list_of_sugarfree_waffles) or not is_available(list_of_sugarfree_waffles)) or is_added_while_preparing(list_of_full_calorie_sweetners, 'waffle'):
                    return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((sugar_free_pancakes_or_waffles_or_french_toast, "Sugar-free pancakes, waffles or French toast"))
       
        ## Add fresh fruits as toppings on sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait ##
        def fresh_fruits_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_fresh_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_fresh_fruits):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fresh_fruits_if_sweet, \
            "Add fresh fruits as toppings on sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait"))
        
        ## Prefer Plain yogurt over sweetened or flavored ##
        def plain_yogurt(list_of_yogurt, list_of_plain_yoghurt, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_plain_yoghurt):
                if not is_used(list_of_plain_yoghurt):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((plain_yogurt, "Prefer Plain yogurt over sweetened or flavored"))
        
        ## Prefer berries as the fruit topping with yoghurt or cereal ##
        def berries_as_topping(list_of_yoghurts, list_of_cereals, list_of_berries, **kwargs):
            if task_goal not in task_list.yoghurt+task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_available(list_of_berries):
                if not is_used(list_of_berries):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((berries_as_topping, "Prefer berries as the fruit topping with yoghurt or cereal"))
        
        ## Prefer eggs with a side of vegetables ##
        def eggs_with_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_vegetables):
                if not is_used(list_of_vegetables):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((eggs_with_vegetables, "Prefer eggs with a side of vegetables"))
        
        ## Add salt and pepper to eggs and vegetables ##
        def eggs_with_salt_and_pepper(list_of_eggs, list_of_vegetables, list_of_salts, list_of_pepper, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not is_available(list_of_salts) or not is_available(list_of_pepper):
                return 'inapplicable'
            if not are_objects_mixed(list_of_salts, list_of_eggs) or not are_objects_mixed(list_of_pepper, list_of_eggs):
                return 'violated'
            if not are_objects_mixed(list_of_salts, list_of_vegetables) or not are_objects_mixed(list_of_pepper, list_of_vegetables):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_with_salt_and_pepper, "Add salt and pepper to eggs and vegetables"))
        
        ## Prefer eggs and vegetables seasoned with no other spices except salt and pepper ##
        def eggs_with_salt_and_pepper_only(list_of_eggs, list_of_vegetables, list_of_spices_except_salt_pepper, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_used(list_of_vegetables):
                if are_objects_mixed(list_of_spices_except_salt_pepper, list_of_eggs) or are_objects_mixed(list_of_spices_except_salt_pepper, list_of_vegetables):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((eggs_with_salt_and_pepper_only, "Prefer eggs and vegetables seasoned with no other spices except salt and pepper"))
        
        ## Use non-stick pans when cooking ##
        def non_stick_pan(list_of_pan, list_of_non_stick_pan, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if is_available(list_of_non_stick_pan):
                if not is_used(list_of_non_stick_pan):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((non_stick_pan, "Use non-stick pans when cooking"))
        
        ## Use silicone spatulas when cooking ##
        def silicone_spatula(list_of_spatula, list_of_silicone_spatula, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if is_available(list_of_silicone_spatula):
                if not is_used(list_of_silicone_spatula):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((silicone_spatula, "Use silicone spatulas when cooking"))

        
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