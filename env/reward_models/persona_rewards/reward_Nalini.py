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

        ## Poach eggs unless a different cooking method is specified ##
        def poach_eggs(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not 'omelette' in task_goal:
                if not (is_action_performed('poach', list_of_eggs) or is_created('poach')):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((poach_eggs, "Poach eggs unless a different cooking method is specified"))
        
        ## Add milk when making scrambled eggs or omelette ##
        def scrambled_eggs_with_milk(list_of_eggs, list_of_milks, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_milks):
                if not are_objects_mixed(list_of_milks, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((scrambled_eggs_with_milk, "Add milk when making scrambled eggs or omelette"))
        
        ## Sprinkle cheese on eggs ##
        def cheese_on_eggs(list_of_eggs, list_of_cheeses, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_cheeses):
                if not are_objects_mixed(list_of_cheeses, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((cheese_on_eggs, "Sprinkle cheese on eggs"))
        
        ## Serve at least two condiments with eggs ##
        def two_condiments_with_eggs(list_of_eggs, list_of_condiments, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if sum([is_available(x) for x in list_of_condiments]) < 2:
                return 'inapplicable'
            if sum([is_used(x) for x in list_of_condiments]) < 2:
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((two_condiments_with_eggs, "Serve at least two condiments with eggs"))
        
        ## Top toast with cheese ##
        def cheese_on_toast(list_of_toasts, list_of_cheeses, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_cheeses):
                if not are_objects_mixed(list_of_cheeses, list_of_toasts):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((cheese_on_toast, "Top toast with cheese"))
        
        ## Use herbs on toast ##
        def herbs_on_toast(list_of_toasts, list_of_herbs, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_herbs):
                if not are_objects_mixed(list_of_herbs, list_of_toasts):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((herbs_on_toast, "Use herbs on toast"))
        
        ## Use dairy milk when using any milk ##
        def dairy_milk(list_of_milks, list_of_dairy_milks, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_available(list_of_dairy_milks):
                if is_available(list_of_dairy_milks) and not is_used(list_of_dairy_milks):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((dairy_milk, "Use dairy milk when using any milk"))
        
        ## Add an extra splash of dairy milk when cooking sweet breakfasts, like cereal, yoghurt parfait, pancakes, french toast, oatmeal ##
        def milk_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_milks, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_available(list_of_milks):
                if is_used(list_of_milks):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((milk_if_sweet, \
            "Add an extra splash of dairy milk when cooking sweet breakfasts, like cereal, yoghurt parfait, pancakes, french toast, oatmeal"))
        
        ## Use fruits as a side for sweet breakfasts, such as cereal, yoghurt parfait, pancakes, french toast, oatmeal ##
        def fruits_as_side_for_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_available(list_of_fruits):
                return 'inapplicable'
            if is_used(list_of_fruits):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fruits_as_side_for_sweet_breakfasts, \
            "Use fruits as a side for sweet breakfasts, such as cereal, yoghurt parfait, pancakes, french toast, oatmeal"))
        
        ## Serve beverages like coffee and tea iced, not hot ##
        def iced_beverages(list_of_coffee_and_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if is_available(list_of_ice):
                if not are_objects_mixed(list_of_ice, list_of_coffee_and_tea):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((iced_beverages, "Serve beverages like coffee and tea iced, not hot"))
        
        ## Use pour over coffee method ##
        def pour_over_coffee(list_of_coffee, list_of_pour_over, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if not is_available(list_of_pour_over):
                return 'inapplicable'
            if not is_used(list_of_pour_over):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((pour_over_coffee, "Use pour over coffee method"))
        
        ## Beverages served black, without any added sugars or creamers ##
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
        self.preference_functions_and_messages.append((black_beverages, "Beverages served black, without any added sugars or creamers"))
        
        ## Use stainless steel or cast iron cookware ##
        def use_stainless_steel_or_cast_iron_cookware(list_of_cookwares, list_of_cast_iron_cookwares, list_of_stainless_steel_cookwares, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if not (is_used(list_of_stainless_steel_cookwares) or is_used(list_of_cast_iron_cookwares)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((use_stainless_steel_or_cast_iron_cookware, "Use stainless steel or cast iron cookware"))

        
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