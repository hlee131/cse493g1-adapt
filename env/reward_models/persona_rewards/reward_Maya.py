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

        ## Add only salt and pepper to eggs aside from oil ##
        def eggs_with_salt_pepper_only(list_of_eggs, list_of_salt, list_of_pepper, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not is_available(list_of_salt) and not is_available(list_of_pepper):
                return 'inapplicable'
            if not are_objects_mixed(list_of_salt, list_of_eggs) and is_available(list_of_salt):
                return 'violated'
            if not are_objects_mixed(list_of_pepper, list_of_eggs) and is_available(list_of_pepper):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((eggs_with_salt_pepper_only, \
            "Add only salt and pepper to eggs aside from oil"))
        
        ## Prefer healthy fats, such as olive oil to cook with ##
        def olive_oil(list_of_cooking_fats, list_of_olive_oil, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if is_available(list_of_olive_oil):
                if not is_used(list_of_olive_oil):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((olive_oil, "Prefer healthy fats, such as olive oil to cook with"))


        ## Sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait, and even cereals, topped with fresh fruits instead of syrup or sugar ##
        def fresh_fruits_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cereals, list_of_fresh_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_fresh_fruits):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((fresh_fruits_if_sweet, \
            "Sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait, and even cereals, topped with fresh fruits instead of syrup or sugar"))

        ## Sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait, and even cereals, topped with nuts ##
        def nuts_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cereals, list_of_nuts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_nuts):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((nuts_if_sweet, \
            "Sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait, and even cereals, topped with nuts"))

        ## Sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait, and even cereals, topped with granola ##
        def granola_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cereals, list_of_granola, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_granola):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((granola_if_sweet, \
            "Sweet breakfasts, such as pancake, waffles, french toast, oatmeal or yoghurt parfait, and even cereals, topped with granola"))

        ## Always add fruits before nuts or granola ##
        def fruits_before_nuts_or_granola(list_of_fruits, list_of_nuts, list_of_granola, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used(list_of_nuts) and are_objects_mixed_in_order(list_of_fruits, list_of_nuts):
                return 'violated'
            if is_used(list_of_granola) and are_objects_mixed_in_order(list_of_fruits, list_of_granola):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((fruits_before_nuts_or_granola, "Always add fruits before nuts or granola"))

        ## Always add nuts before granola ##
        def nuts_before_granola(list_of_nuts, list_of_granola, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used(list_of_granola) and are_objects_mixed_in_order(list_of_nuts, list_of_granola):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((nuts_before_granola, "Always add nuts before granola"))

        ## Always use dairy-based milks alternatives, and avoid plant-based milks ##
        def no_plant_based_milks(list_of_milks, list_of_non_plant_based_milk, list_of_plant_based_milks, **kwargs):
            if task_goal not in task_list.beverages+task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_available(list_of_non_plant_based_milk):
                if is_used(list_of_plant_based_milks):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((no_plant_based_milks, "Always use dairy-based milks alternatives, and avoid plant-based milks"))
        
        ## Always use dairy-based yoghurts alternatives, and avoid plant-based yoghurts ##
        def no_plant_based_yogurt(list_of_yogurt, list_of_plant_based_yoghurt, list_of_dairy_based_yoghurts, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_dairy_based_yoghurts):
                if is_used(list_of_plant_based_yoghurt):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((no_plant_based_yogurt, "Always use dairy-based yoghurts alternatives, and avoid plant-based yoghurts"))
        
        ## Beverages like tea and coffee are preferred hot, not iced ##
        def hot_tea(list_of_coffee_and_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if are_objects_mixed(list_of_ice, list_of_coffee_and_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((hot_tea, "Beverages like tea and coffee are preferred hot, not iceds"))
        
        ## Use the espresso machine, if available, else pour over to make coffee ##
        def espresso_machine(list_of_coffee, list_of_espresso_machine, list_of_pour_over, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if is_available(list_of_espresso_machine):
                if not is_used(list_of_espresso_machine):
                    return 'violated'
            elif is_available(list_of_pour_over):
                if not is_used(list_of_pour_over):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((espresso_machine, "Use the espresso machine, if available, else pour over to make coffee"))
        
        ## Serve beverages such as tea and coffee with milk next to it, but not mixed in ##
        def milk_next_to_beverage(list_of_coffee_and_tea, list_of_milks, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if are_objects_mixed(list_of_milks, list_of_coffee_and_tea):
                return 'violated'
            if not is_used(list_of_milks):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((milk_next_to_beverage, \
            "Serve beverages such as tea and coffee with milk next to it, but not mixed in"))

        ## Have breakfast on the kitchen island ##
        def have_breakfast_on_kitchen_island(list_of_kitchen_island, **kwargs):
            if all_final_food_location(list_of_kitchen_island):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((have_breakfast_on_kitchen_island, "Have breakfast on the kitchen island"))
        
        
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