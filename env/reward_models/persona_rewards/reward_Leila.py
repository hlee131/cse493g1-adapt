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

        ## Toast with sweet spreads like jams, marmalades, and nut butters ##
        def toast_with_spreads(list_of_bread, list_of_spreads, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not are_objects_mixed(list_of_spreads, list_of_bread):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((toast_with_spreads, \
             "Toast with sweet spreads like jams, marmalades, and nut butters"))
        
        ## Add honey or maple syrup to sweet breakfasts, like oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def sweet_breakfasts_with_syrup(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_cereal, list_of_yoghurts, list_of_syrup, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_syrup):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sweet_breakfasts_with_syrup, \
            "Add honey or maple syrup to sweet breakfasts, like oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
            
        ## Add fresh fruit to sweet breakfasts, like french toast, cereal, yoghurt parfait, oatmeal, pancakes ##
        def sweet_breakfasts_with_fruit(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_cereal, list_of_yoghurts, list_of_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_fruits):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sweet_breakfasts_with_fruit, \
            "Add fresh fruit to sweet breakfasts, like french toast, cereal, yoghurt parfait, oatmeal, pancakes"))
        
        ## Add nuts to sweet breakfasts, like french toast, cereal, yoghurt parfait, oatmeal, pancakes ##
        def sweet_breakfasts_with_nuts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cereal, list_of_nuts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_nuts):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sweet_breakfasts_with_nuts, \
            "Add nuts to sweet breakfasts, like french toast, cereal, yoghurt parfait, oatmeal, pancakes"))
        
        ## Add granola to yoghurt if available ##
        def flavored_yoghurt_with_granola_or_fruit(list_of_yoghurts, list_of_granola, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_granola):
                if not are_objects_mixed(list_of_granola, list_of_yoghurts):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((flavored_yoghurt_with_granola_or_fruit, "Add granola to yoghurt if available"))
        
        ## Savory items served with a side of sautéed vegetables or hash browns ##
        def breakfast_meats_with_sides(list_of_savory_ingredients, list_of_vegetables, list_of_hash_browns, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if not is_available(list_of_vegetables) and not is_available(list_of_hash_browns):
                return 'inapplicable'
            if not is_used(list_of_vegetables):
                if not is_used(list_of_hash_browns):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((breakfast_meats_with_sides, "Savory items served with a side of sautéed vegetables or hash browns"))
        
        ## Beverages like tea and coffee served hot ##
        def hot_beverages(list_of_coffee_and_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if are_objects_mixed(list_of_ice, list_of_coffee_and_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((hot_beverages, "Beverages like tea and coffee served hot"))
        
        ## Beverages like tea and coffee served with honey, when available or else a bit of sugar ##
        def bev_with_honey(list_of_coffee_and_tea, list_of_honey, list_of_sweetners, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if not is_available(list_of_sweetners):
                return 'inapplicable'
            if is_available(list_of_honey):
                if not are_objects_mixed(list_of_honey, list_of_coffee_and_tea):
                    return 'violated'
            elif not are_objects_mixed(list_of_sweetners, list_of_coffee_and_tea):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((bev_with_honey, "Beverages like tea and coffee served with honey, when available or else a bit of sugar"))
        
        ## Tea served with lemon ##
        def tea_with_lemon(list_of_tea, list_of_lemon, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if is_available(list_of_lemon):
                if not are_objects_mixed(list_of_lemon, list_of_tea):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((tea_with_lemon, "Tea served with lemon"))
        
        ## Melted cheese is a staple in her diet, especially when paired with eggs and veggies ##
        def melted_cheese_with_eggs_and_veggies(list_of_cheese, list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_cheese):
                if not are_objects_mixed(list_of_eggs, list_of_cheese):
                    if not are_objects_mixed(list_of_vegetables, list_of_cheese):
                        return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((melted_cheese_with_eggs_and_veggies, "Melted cheese is a staple in her diet, especially when paired with eggs and veggies"))

        ## Add at least two spices to savory breakfast dishes ##
        def savory_breakfast_with_spices(list_of_savory_ingredients, list_of_spices, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if not are_objects_mixed(list_of_spices, list_of_savory_ingredients) and is_available(list_of_spices):
                return 'violated'
            if (sum([is_used(spice) for spice in list_of_spices]) < 2) and (sum([is_available(spice) for spice in list_of_spices]) >= 2):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((savory_breakfast_with_spices, "Add at least two spices to savory breakfast dishes"))
        
        ## Add herbs to savory breakfast dishes ##
        def savory_breakfast_with_herbs(list_of_savory_ingredients, list_of_herbs, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_herbs):
                if not are_objects_mixed(list_of_herbs, list_of_savory_ingredients):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((savory_breakfast_with_herbs, "Add herbs to savory breakfast dishes"))
        
        ## Add a side of sauteed vegetables to savory breakfast ##
        def savory_breakfast_with_veggies(list_of_savory_ingredients, list_of_vegetables, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_vegetables):
                if not is_used(list_of_vegetables):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((savory_breakfast_with_veggies, "Add a side of sauteed vegetables to savory breakfast"))
        
        ## Use cast-iron cookware for breakfast dishes ##
        def cast_iron_cookware(list_of_cookware, list_of_cast_iron_cookwares, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if is_available(list_of_cast_iron_cookwares):
                if not is_used(list_of_cast_iron_cookwares):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((cast_iron_cookware, "Use cast-iron cookware for breakfast dishes"))
        
        ## Use expensive glassware for breakfast ##
        def fancy_dishes(list_of_dishes, list_of_fragile_expensive_kitchenwares, **kwargs):
            if is_available(list_of_fragile_expensive_kitchenwares):
                if not is_used(list_of_fragile_expensive_kitchenwares):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((fancy_dishes, "Use expensive glassware for breakfast"))

        ## Prefers a small dessert of fresh fruit and whipped cream with savory breakfasts ##
        def surprise_breakfast_dessert(list_of_fresh_fruits, list_of_whipped_cream, list_of_eggs, **kwargs):
            if not is_available(list_of_fresh_fruits) and not is_available(list_of_whipped_cream):
                return 'inapplicable'
            if is_used(list_of_eggs):
                if (is_used(list_of_fresh_fruits) or not is_available(list_of_fresh_fruits)) and (is_used(list_of_whipped_cream) or not is_available(list_of_whipped_cream)):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((surprise_breakfast_dessert,"Prefers a small dessert of fresh fruit and whipped cream with savory breakfasts"))
        
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
        