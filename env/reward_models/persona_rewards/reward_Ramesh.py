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

        ## If eggs are eaten, make them into an omelette with at least 3 spices ##
        def omelette_with_spices_and_vegetables(list_of_eggs, list_of_spices, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not (is_action_performed('make omelette', list_of_eggs) or is_created('omelette') or is_created('omelet')):
                return 'violated'
            if is_available(list_of_spices) and not are_objects_mixed(list_of_spices, list_of_eggs):
                return 'violated'
            if sum([is_used(x) for x in list_of_spices]) < 3:
                if sum([is_available(x) for x in list_of_spices]) >= 3:
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((omelette_with_spices_and_vegetables, "If eggs are eaten, make them into an omelette with at least 3 spices"))
        
        ## Add at least two vegetables when making eggs ##
        def eggs_with_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if sum([is_available(x) for x in list_of_vegetables]) >= 2:
                if sum([is_used(x) for x in list_of_vegetables]) < 2:
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((eggs_with_vegetables, "Add at least two vegetables when making eggs"))
            
        ## Prefer using butter, for example on toast ##
        def butter_on_toast(list_of_toast, list_of_butter, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_butter):
                if not are_objects_mixed(list_of_butter, list_of_toast):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((butter_on_toast, "Prefer using butter, for example on toast"))
        
        ## Prefer using butter, for example when cooking eggs ##
        def butter_with_eggs(list_of_butter, list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_butter):
                if not are_objects_mixed(list_of_butter, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((butter_with_eggs, "Prefer using butter, for example when cooking eggs"))
            
        ## Prefer beverages like tea and coffee served hot ##
        def hot_bevs(list_of_tea, list_of_coffee, list_of_ice, **kwargs):
            if task_goal not in task_list.tea+task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if are_objects_mixed(list_of_ice, list_of_tea) or are_objects_mixed(list_of_ice, list_of_coffee):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((hot_bevs, "Prefer beverages like tea and coffee served hot"))
            
        ## Prefer beverages like tea and coffee served hot with milk and sugar ##
        def bevs_with_milk_and_sugar(list_of_tea, list_of_coffee, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.tea+task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if task_goal in task_list.tea:
                if (is_available(list_of_milk) and not are_objects_mixed(list_of_milk, list_of_tea)) or (is_available(list_of_sugar) and not are_objects_mixed(list_of_sugar, list_of_tea)):
                    return 'violated'
            if task_goal in task_list.coffee:
                if (is_available(list_of_milk) and not are_objects_mixed(list_of_milk, list_of_coffee)) or (is_available(list_of_sugar) and not are_objects_mixed(list_of_sugar, list_of_coffee)):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((bevs_with_milk_and_sugar, "Prefer beverages like tea and coffee served hot with milk and sugar"))
        
        ## Prefer a hint of cardamom in tea and coffee ##
        def cardamom_in_bevs(list_of_tea, list_of_coffee, list_of_cardamom, **kwargs):
            if task_goal not in task_list.tea+task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if task_goal in task_list.tea:
                if (is_available(list_of_cardamom) and not are_objects_mixed(list_of_cardamom, list_of_tea)):
                    return 'violated'
            if task_goal in task_list.coffee:
                if (is_available(list_of_cardamom) and not are_objects_mixed(list_of_cardamom, list_of_coffee)):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((cardamom_in_bevs, "Prefer a hint of cardamom in tea and coffee"))
        
        ## If yogurt is eaten, prefer it plain ##
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
        self.preference_functions_and_messages.append((plain_yogurt, "If yogurt is eaten, prefer it plain"))
        
        ## Add a hint of cardamom to sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait ##
        def cardamom_in_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cardamom, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_cardamom):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((cardamom_in_sweet_breakfasts, "Add a hint of cardamom to sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait"))
        
        ## Add nuts as a topping to sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait ##
        def nuts_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_nuts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_nuts):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((nuts_on_sweet_breakfasts, "Add nuts as a topping to sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait"))
        
        ## Top eggs and toast with cheese ##
        def cheese_on_eggs_and_toast(list_of_eggs, list_of_toast, list_of_cheese, **kwargs):
            if task_goal not in task_list.eggs+task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_cheese):
                if not are_objects_mixed(list_of_cheese, list_of_eggs) and not are_objects_mixed(list_of_cheese, list_of_toast):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((cheese_on_eggs_and_toast, "Top eggs and toast with cheese"))
        
        ## Use chili as a garnish or provide hot sauce as a condiment ##
        def chili_or_hot_sauce(list_of_eggs, list_of_toast, list_of_chili_flakes, list_of_hot_sauces, **kwargs):
            if task_goal not in task_list.eggs+task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not is_available(list_of_chili_flakes) and not is_available(list_of_hot_sauces):
                return 'inapplicable'
            if not are_objects_mixed(list_of_chili_flakes, list_of_eggs) and not are_objects_mixed(list_of_chili_flakes, list_of_toast):
                if not is_used(list_of_hot_sauces):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((chili_or_hot_sauce, "Use chili as a garnish or provide hot sauce as a condiment"))
        
        ## Fresh herbs like cilantro or mint make a lovely garnish for eggs ##
        def fresh_herbs_as_garnish(list_of_eggs, list_of_cilantro, list_of_mint, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_cilantro) or is_available(list_of_mint):
                if are_objects_mixed(list_of_cilantro, list_of_eggs) or are_objects_mixed(list_of_mint, list_of_eggs):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((fresh_herbs_as_garnish, "Fresh herbs like cilantro or mint make a lovely garnish for eggs"))
        
        ## Serve breakfast at the formal dining room or outdoor patio ##
        def serve_at_formal_dining_room_or_outdoor_patio(list_of_dining_room_tables, list_of_patio_tables, **kwargs):
            if all_final_food_location(list_of_dining_room_tables) or all_final_food_location(list_of_patio_tables):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((serve_at_formal_dining_room_or_outdoor_patio, "Serve breakfast at the formal dining room or outdoor patio"))


        
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
                penalty += 1
                max_penalty += 1
                messages_task.append(f"FAILED: {preference_response}")
                penalty += 1
                max_penalty += 1
                messages.append(f"FAILED: {message}")
                 
        self.penalty = penalty
        self.max_penalty = max_penalty
        # self.messages = list(set(messages_task)) + messages
        self.messages = messages