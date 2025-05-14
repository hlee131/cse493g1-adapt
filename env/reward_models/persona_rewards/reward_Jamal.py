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
        
        ## Eggs should be accompanied by crispy bacon ##
        def eggs_with_bacon(list_of_eggs, list_of_bacon, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_bacon):
                if not is_used(list_of_bacon):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((eggs_with_bacon,"Eggs should be accompanied by crispy bacon"))
        
        ## Eggs should be accompanied by toast ##
        def eggs_with_toast(list_of_eggs, list_of_breads, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_breads):
                if not is_used(list_of_breads):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((eggs_with_toast,"Eggs should be accompanied by toast"))

        ## Only use traditional full calorie sugars in sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait ##
        def traditional_full_calorie_sugars(list_of_sweetners, list_of_full_calorie_sweetners, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_available(list_of_full_calorie_sweetners):
                if not is_used(list_of_full_calorie_sweetners):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((traditional_full_calorie_sugars,"Only use traditional full calorie sugars in sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait"))
        
        ## Only use full fat dairy milk ##
        def full_fat_dairy_milk(list_of_milks, list_of_skim_milks, list_of_non_dairy_milks, list_of_full_fat_milk, **kwargs):
            if task_goal not in task_list.cereal+task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if is_available(list_of_full_fat_milk):
                if is_used(list_of_skim_milks):
                    return 'violated'
                if is_used(list_of_non_dairy_milks):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((full_fat_dairy_milk,"Only use full fat dairy milk"))
        
        ## Oatmeal cooked with milk ##
        def oatmeal_with_milk(list_of_oatmeal, list_of_milk, **kwargs):
            if task_goal not in task_list.oatmeal:
                return 'inapplicable'
            if task_goal in task_list.oatmeal and not task_check.check(task_list.oatmeal):
                return task_check.message(task_goal, task_list.oatmeal)
            if is_available(list_of_milk):
                if not are_objects_mixed(list_of_milk, list_of_oatmeal):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((oatmeal_with_milk,"Oatmeal cooked with milk"))
        
        ## Add a touch of sweetness to non-savory breakfasts such as pancakes, waffles, oatmeal, yoghurt parfait, etc. ##
        def non_savory_with_sweetness(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_sweetners, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_sweetners):
                return 'satisfied'
            return 'violated' 
        self.preference_functions_and_messages.append((non_savory_with_sweetness,"Add a touch of sweetness to non-savory breakfasts such as pancakes, waffles, oatmeal, yoghurt parfait, etc."))
        
        ## Sweet breakfasts like french toast, cereal, yoghurt parfait, oatmeal, pancakes should be topped with maple syrup or honey ##
        def pancakes_and_french_toast_with_syrup(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_syrups, list_of_honey, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_syrups+list_of_honey):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((pancakes_and_french_toast_with_syrup,"Sweet breakfasts like french toast, cereal, yoghurt parfait, oatmeal, pancakes should be topped with maple syrup or honey"))
       
        ## Sweet breakfasts french toast, cereal, yoghurt parfait, oatmeal, pancakes topped with nuts ##
        def pancakes_and_french_toast_with_nut(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_nuts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_nuts):
                return 'satisfied'
            return 'violated' 
        self.preference_functions_and_messages.append((pancakes_and_french_toast_with_nut,"Sweet breakfasts french toast, cereal, yoghurt parfait, oatmeal, pancakes topped with nuts"))
        
        ## Prefer flavored yoghurt ##
        def flavored_yoghurt(list_of_yoghurts, list_of_flavored_yoghurts, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_flavored_yoghurts):
                if not is_used(list_of_flavored_yoghurts):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((flavored_yoghurt,"Prefer flavored yoghurt"))
        
        ## Yoghurt mixed with granola ##
        def yoghurt_with_granola(list_of_yoghurts, list_of_granola, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_granola):
                if are_objects_mixed(list_of_granola, list_of_yoghurts):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'    
        self.preference_functions_and_messages.append((yoghurt_with_granola,"Yoghurt mixed with granola"))
        
        ## Beverages such as tea and coffee preferred hot ##
        def bevs_hot(list_of_tea, list_of_coffee, list_of_ice, **kwargs):
            if task_goal not in task_list.tea+task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.tea:
                if task_goal in task_list.tea and not task_check.check(task_list.tea):
                    return task_check.message(task_goal, task_list.tea)
                if are_objects_mixed(list_of_ice, list_of_tea):
                    return 'violated'
            if task_goal in task_list.coffee:
                if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                    return task_check.message(task_goal, task_list.coffee)
                if are_objects_mixed(list_of_ice, list_of_coffee):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((bevs_hot,"Beverages such as tea and coffee preferred hot"))
        
        ## Beverages such as tea and coffee preferred with milk and sugar ##
        def bevs_with_milk_and_sugar(list_of_tea, list_of_coffee, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.tea+task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.tea:
                if task_goal in task_list.tea and not task_check.check(task_list.tea):
                    return task_check.message(task_goal, task_list.tea)
                if not are_objects_mixed(list_of_milk, list_of_tea) and is_available(list_of_milk):
                    return 'violated'
                if not are_objects_mixed(list_of_sugar, list_of_tea) and is_available(list_of_sugar):
                    return 'violated'
            if task_goal in task_list.coffee:
                if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                    return task_check.message(task_goal, task_list.coffee)
                if not are_objects_mixed(list_of_milk, list_of_coffee) and is_available(list_of_milk):
                    return 'violated'
                if not are_objects_mixed(list_of_sugar, list_of_coffee) and is_available(list_of_sugar):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((bevs_with_milk_and_sugar,"Beverages such as tea and coffee preferred with milk and sugar"))
        
        ## Melted cheese is a favorite addition to savory breakfast dishes, such as eggs ##
        def melted_cheese_with_scrambled_eggs(list_of_cheese, list_of_savory_ingredients, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_cheese):
                if not is_used(list_of_cheese):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((melted_cheese_with_scrambled_eggs,"Melted cheese is a favorite addition to savory breakfast dishes, such as eggs"))
        
        ## A sprinkle of chopped parsley or chives would be a nice touch to eggs ##
        def parsley_chives(list_of_fresh_herbs, list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_fresh_herbs):
                if is_used(list_of_fresh_herbs):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((parsley_chives,"A sprinkle of chopped parsley or chives would be a nice touch to eggs"))
        
        ## Serving beverages first would be a good approach, so it is ready to be had as soon as possible ##
        def prepare_beverages_first(list_of_beverages, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if not is_served_before(list_of_beverages, list_of_food):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((prepare_beverages_first,"Serving beverages first would be a good approach, so it is ready to be had as soon as possible"))
        
        ## Use butter to cook eggs and top toast ##
        def butter_with_eggs_and_toast(list_of_eggs, list_of_toast, list_of_butter, **kwargs):
            if task_goal not in task_list.eggs+task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not (are_objects_mixed(list_of_butter, list_of_eggs) or are_objects_mixed(list_of_butter, list_of_toast)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((butter_with_eggs_and_toast,"Use butter to cook eggs and top toast"))
        
        ## Prepare one of sausage, hash browns, or toast as a side when making eggs ##
        def complement_breakfast_choices(list_of_eggs, list_of_sausages, list_of_hash_browns, list_of_toast, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_used_if_available(list_of_sausages+list_of_hash_browns+list_of_toast):
                return 'satisfied'
            if is_created('sausage') or is_created('toast') or is_created('hash brown'):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((complement_breakfast_choices,"Prepare one of sausage, hash browns, or toast as a side when making eggs"))
        
        ## Prefer to have a condiment as a side with savory breakfasts ##
        def condiment_with_savory(list_of_savory_ingredients, list_of_condiments, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_condiments):
                if not is_used(list_of_condiments):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((condiment_with_savory,"Prefer to have a condiment as a side with savory breakfasts"))
        
        ## Do not put vegetables in savory breakfasts ##
        def no_veggies_in_savory_breakfasts(list_of_savory_ingredients, list_of_vegetables, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_used(list_of_vegetables):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_veggies_in_savory_breakfasts,"Do not put vegetables in savory breakfasts"))

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