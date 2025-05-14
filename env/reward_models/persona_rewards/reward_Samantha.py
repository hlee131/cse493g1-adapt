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

        ## Eggs scrambled or made into an omelette 
        def eggs_scrambled_or_omelette(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not (is_action_performed('scramble', list_of_eggs) or is_action_performed('make omelette', list_of_eggs) or is_created('scramble') or is_created('omelette')):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_scrambled_or_omelette, "Eggs scrambled or made into an omelette"))
        
        ## Eggs with vegetables like bell peppers, onions, and mushrooms ##
        def eggs_with_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_vegetables, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_with_vegetables, "Eggs with vegetables like bell peppers, onions, and mushrooms"))
       
        ## Prefer side of fruits or avocado with eggs ##
        def side_of_fruits_or_avocado_with_eggs(list_of_eggs, list_of_fruits, list_of_avocado, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not is_available(list_of_fruits) and not is_available(list_of_avocado):
                return 'inapplicable'
            if not (is_used(list_of_fruits) or is_used(list_of_avocado)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((side_of_fruits_or_avocado_with_eggs, "Prefer side of fruits or avocado with eggs"))
       
        ## Toast toasted with spreads like avocado or hummus; instead of nut-based spreads ##
        def toast_with_spreads(list_of_toast, list_of_avocado, list_of_hummus, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not (are_objects_mixed(list_of_avocado, list_of_toast) or are_objects_mixed(list_of_hummus, list_of_toast)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((toast_with_spreads, "Toast toasted with spreads like avocado or hummus; instead of nut-based spreads"))
        
        ## add butter on toast ##
        def add_butter_on_toast(list_of_toast, list_of_butter, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if are_objects_mixed(list_of_butter, list_of_toast):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((add_butter_on_toast, "add butter on toast"))
        
        ## Prefer herbal teas like peppermint or chamomile ##
        def herbal_tea(list_of_tea, list_of_peppermint_tea, list_of_chamomile_tea, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if not (is_used(list_of_peppermint_tea) or is_used(list_of_chamomile_tea)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((herbal_tea, "Prefer herbal teas like peppermint or chamomile"))
        
        ## Beverages such as tea or coffee should be black, without milk or sugar
        def black_coffee(list_of_coffee, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if are_objects_mixed(list_of_milk, list_of_coffee) or are_objects_mixed(list_of_sugar, list_of_coffee):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((black_coffee, \
            "Beverages such as tea or coffee should be black, without milk or sugar"))
        
        ## Avoid nuts and nut based spreads on toast ##
        def no_nuts(list_of_nut_butter, list_of_nuts, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_used(list_of_nut_butter) or is_used(list_of_nuts):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_nuts, "Avoid nuts and nut based spreads on toast"))
        
        ## When adding milk to cereal, avoid nut-based milk, use a dairy-based alternative instead ##
        def no_nut_milk(list_of_milk, list_of_nut_milk, list_of_milks, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_used(list_of_nut_milk):
                return 'violated'
            if not is_used_if_available(list_of_milks):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_nut_milk, "When adding milk to cereal, avoid nut-based milk, use a dairy-based alternative instead"))
        
        ## When using yoghurt, avoid nut-based yoghurt ##
        def no_nut_yoghurt(list_of_yoghurts, list_of_nut_yoghurt, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_used(list_of_nut_yoghurt):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_nut_yoghurt, "When using yoghurt, avoid nut-based yoghurt"))
        
        ## Add fruits as a topping to sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def fruits_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_fruits):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fruits_on_sweet_breakfasts, "Add fruits as a topping to sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Serve items in a certain order, starting with the food first, then beverages ##
        def bring_out_items_in_order(list_of_coffee_and_tea, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if is_used(list_of_food):
                if not is_served_before(list_of_food, list_of_coffee_and_tea):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((bring_out_items_in_order, "Serve items in a certain order, starting with the food first, then beverages"))
        
        ## Eat breakfast at a standing desk or kitchen island##
        def eat_at_standing_desk_or_kitchen_island(list_of_standing_desk, list_of_kitchen_island, **kwargs):
            if all_final_food_location(list_of_standing_desk) or all_final_food_location(list_of_kitchen_island):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((eat_at_standing_desk_or_kitchen_island, "Eat breakfast at a standing desk or kitchen island"))


        
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