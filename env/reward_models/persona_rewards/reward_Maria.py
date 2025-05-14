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

        ## Prefer eggs made into an omelette, unless otherwise specified ##
        def eggs_as_omelette(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_created('omelette') or is_created('omelet'):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((eggs_as_omelette, "Prefer eggs made into an omelette, unless otherwise specified"))
        
        ## Add in some veggies like bell peppers or spinach to eggs ##
        def eggs_with_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_vegetables, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_with_vegetables, "Add in some veggies like bell peppers or spinach to eggs"))
        
        ## My toast is topped with nut butter or avocado spread ##
        def toast_with_nut_butter_or_avocado(list_of_toast, list_of_nut_butter, list_of_avocado, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not is_used_if_available(list_of_nut_butter+list_of_avocado):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((toast_with_nut_butter_or_avocado, "My toast is topped with nut butter or avocado spread"))
        
        ## Coffee should be strong and iced ##
        def iced_coffee(list_of_coffee, list_of_ice, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if is_available(list_of_ice):
                if not are_objects_mixed(list_of_ice, list_of_coffee):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((iced_coffee, "Coffee should be strong and iced"))
        
        ## Top pancakes or French toast with fresh fruit or a drizzle of honey ##
        def pancakes_or_french_toast_with_fruit_or_honey(list_of_pancakes, list_of_french_toast, list_of_fruit, list_of_honey, **kwargs):
            if task_goal not in task_list.pancakes:
                return 'inapplicable'
            if task_goal in task_list.pancakes and not task_check.check(task_list.pancakes):
                return task_check.message(task_goal, task_list.pancakes)
            if not is_used_if_available(list_of_fruit+list_of_honey):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((pancakes_or_french_toast_with_fruit_or_honey, "Top pancakes or French toast with fresh fruit or a drizzle of honey"))
        
        ## Yogurt must be topped with granola ##
        def yogurt_with_granola(list_of_yogurt, list_of_granola, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if not is_used_if_available(list_of_granola):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((yogurt_with_granola, "Yogurt must be topped with granola"))
        
        ## Incorporate foods like nuts, and seeds in sweet breakfasts, such as french toast, cereal, yoghurt parfait, oatmeal, pancakes ##
        def nuts_and_seeds_in_sweet_breakfasts(list_of_nuts, list_of_seeds, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_nuts+list_of_seeds):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((nuts_and_seeds_in_sweet_breakfasts, "Incorporate foods like nuts, and seeds in sweet breakfasts, such as french toast, cereal, yoghurt parfait, oatmeal, pancakes"))
        
        ## Add melted cheese to eggs, so it's easier to digest and adds a rich, creamy flavor ##
        def melted_cheese_with_eggs(list_of_eggs, list_of_cheese, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_cheese, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((melted_cheese_with_eggs, "Add melted cheese to eggs, so it's easier to digest and adds a rich, creamy flavor"))
        
        ## Serve beverages first before foods ##
        def prepare_beverages_first(list_of_beverages, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                    return task_check.message(task_goal, task_list.beverages)
                if not is_served_before(list_of_beverages, list_of_food):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((prepare_beverages_first, "Serve beverages first before foods"))
        
        ## When mixing cereal and milk, I prefer to pour the cereal in first - it helps prevent spills ##
        def cereal_in_first(list_of_cereal, list_of_milk, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if not are_objects_mixed_in_order(list_of_cereal, list_of_milk):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((cereal_in_first, "When mixing cereal and milk, I prefer to pour the cereal in first - it helps prevent spills"))
        
        ## A side of fresh fruit or nuts with savory breakfast is appreciated to satisfy my sweet tooth ##
        def side_of_fruit_or_nuts(list_of_fruit, list_of_nuts, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if not is_used_if_available(list_of_fruit+list_of_nuts):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((side_of_fruit_or_nuts, "A side of fresh fruit or nuts with savory breakfast is appreciated to satisfy my sweet tooth"))
        
        ## Serve breakfast at my desk or on the outdoor patio ##
        def breakfast_at_desk_or_patio(list_of_standing_desk, list_of_patio_tables, **kwargs):
            if all_final_food_location(list_of_standing_desk+list_of_patio_tables):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((breakfast_at_desk_or_patio, "Serve breakfast at my desk or on the outdoor patio"))

        
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