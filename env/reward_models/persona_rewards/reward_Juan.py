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
        
        ## Prefer hearty scrambled eggs or omelettes fitting a high-protein high-calorie diet ##
        def scrambled_or_omelette_eggs(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not (is_action_performed('scramble', list_of_eggs) or is_action_performed('make omelette', list_of_eggs) or is_created('scramble') or is_created('omelette')):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((scrambled_or_omelette_eggs, \
            "Prefer hearty scrambled eggs or omelettes fitting a high-protein high-calorie diet"))
        
        ## Add whole milk while preparing eggs ##
        def whole_milk_with_eggs(list_of_eggs, list_of_milk, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_milk):
                if not are_objects_mixed(list_of_milk, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((whole_milk_with_eggs, \
            "Add whole milk while preparing eggs"))
        
        ## Beverages such as coffee should be black, without milk or sugar
        def black_coffee(list_of_coffee, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if are_objects_mixed(list_of_milk, list_of_coffee) or are_objects_mixed(list_of_sugar, list_of_coffee):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((black_coffee, \
            "Beverages such as coffee should be black, without milk or sugar"))
        
        ## Beverages such as tea should be black, without milk or sugar
        def black_tea(list_of_tea, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if are_objects_mixed(list_of_milk, list_of_tea) or are_objects_mixed(list_of_sugar, list_of_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((black_tea, \
            "Beverages such as tea should be black, without milk or sugar"))
        
        ## Beverages such as coffee should be served iced
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
        self.preference_functions_and_messages.append((iced_coffee, \
            "Beverages such as coffee should be served iced"))
        
        ## Beverages such as tea should be served iced
        def iced_tea(list_of_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if is_available(list_of_ice):
                if not are_objects_mixed(list_of_ice, list_of_tea):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((iced_tea, \
            "Beverages such as tea should be served iced"))
        
        ## Use full-fat milk or whipped cream to maintain a high-fat high-calorie diet ##
        def full_fat_milk(list_of_milk, list_of_full_fat_milk, list_of_whipped_cream, **kwargs):
            if task_goal not in task_list.eggs+task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if not is_available(list_of_full_fat_milk+list_of_whipped_cream):
                return 'inapplicable'
            if not is_used(list_of_milk+list_of_whipped_cream):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((full_fat_milk, \
            "Use full-fat milk or whipped cream to maintain a high-fat high-calorie diet"))
        
        ## Top sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait, with whipped cream to add some extra calories ##
        def sweet_breakfasts_with_whipped_cream(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_whipped_cream, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if not is_used_if_available(list_of_whipped_cream):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sweet_breakfasts_with_whipped_cream, \
            "Top sweet breakfasts, such as french toast, cereal, pancakes, oatmeal, yoghurt parfait, with whipped cream to add some extra calories"))
        
        ## Add cheese or avocado to savory breakfasts to add some extra calories ##
        def add_cheese_or_avocado(list_of_savory_ingredients, list_of_cheese, list_of_avocado, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if not is_available(list_of_cheese+list_of_avocado):
                return 'inapplicable'
            if not is_used(list_of_cheese+list_of_avocado):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((add_cheese_or_avocado, \
            "Add cheese or avocado to savory breakfasts to add some extra calories"))
        
        ## Stainless steel or cast iron cookware and durable utensils
        def stainless_steel_or_cast_iron_cookware(list_of_cookware, list_of_stainless_steel_cookware, list_of_cast_iron_cookware, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if not is_available(list_of_stainless_steel_cookware+list_of_cast_iron_cookware):
                return 'inapplicable'
            if not is_used(list_of_stainless_steel_cookware+list_of_cast_iron_cookware):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((stainless_steel_or_cast_iron_cookware, \
            "Stainless steel or cast iron cookware and durable utensils"))
        
        ## Cream or avocado are great sides when making eggs ##
        def cream_or_avocado_sides(list_of_eggs, list_of_cream, list_of_avocado, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not is_available(list_of_cream+list_of_avocado):
                return 'inapplicable'
            if is_used(list_of_cream+list_of_avocado):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((cream_or_avocado_sides, \
            "Cream or avocado are great sides when making eggs"))
        
        ## Dining room or patio dining locations
        def dining_room_or_patio_dining_locations(list_of_dining_room_table, list_of_patio_table, **kwargs):
            if all_final_food_location(list_of_dining_room_table) or all_final_food_location(list_of_patio_table):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((dining_room_or_patio_dining_locations, \
            "Dining room or patio dining locations"))


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