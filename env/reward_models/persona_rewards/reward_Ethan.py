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

        ## Prefer gluten-free cereals
        def gluten_free_cereals(list_of_cereals, list_of_gluten_free_cereals, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_available(list_of_gluten_free_cereals):
                if not is_used(list_of_gluten_free_cereals):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((gluten_free_cereals, "Prefer gluten-free cereals"))

        ## Prefer gluten-free alternatives, when using bread for savory toast or french toast ##
        def gluten_free_bread_for_toast(list_of_bread, list_of_gluten_free_bread, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_gluten_free_bread):
                if not is_used(list_of_gluten_free_bread):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((gluten_free_bread_for_toast, \
            "Prefer gluten-free alternatives, when using bread for savory toast or french toast"))

        ## Prefer lactose-free milk alternatives ##
        def lactose_free_milk(list_of_milks, list_of_milks_containing_lactose, list_of_plant_based_milks, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_available(list_of_plant_based_milks):
                if is_used(list_of_milks_containing_lactose):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((lactose_free_milk, "Prefer lactose-free milk alternatives"))

        ## Prefer toast without any spreads ##
        def bread_toasted_without_any_spreads(list_of_bread, list_of_spreads, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if are_objects_mixed(list_of_spreads, list_of_bread):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((bread_toasted_without_any_spreads, "Prefer toast without any spreads"))

        ## Do not use butter on toast, when cooking eggs, etc. since it is not lactose-free ##
        def no_butter(list_of_butter, list_of_breads, list_of_savory_ingredients, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if not is_available(list_of_butter):
                return 'inapplicable'
            if is_used(list_of_butter):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_butter, "Do not use butter on toast, when cooking eggs, etc. since it is not lactose-free"))

        ## Sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast etc. are preferred without any sugar, syrups or spreads ##
        def no_toppings_for_pancakes_or_french_toast(list_of_pancakes, list_of_french_toast, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_sweetners, list_of_spreads, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used(list_of_sweetners):
                return 'violated'
            if is_used(list_of_spreads):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_toppings_for_pancakes_or_french_toast, "Sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast etc. are preferred without any sugar, syrups or spreads"))

        ## Prefer yogurt alternatives made from non-dairy sources to avoid lactose ##
        def nondairy_yogurt_alternatives(list_of_yogurt, list_of_yoghurts_containing_lactose, list_of_plant_based_yoghurt, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_plant_based_yoghurt):
                if is_used(list_of_yoghurts_containing_lactose):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((nondairy_yogurt_alternatives, "Prefer yogurt alternatives made from non-dairy sources to avoid lactose"))

        ## Prefer beverages such as coffee served hot ##
        def coffee_served_hot(list_of_coffee, list_of_ice, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if are_objects_mixed(list_of_ice, list_of_coffee):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((coffee_served_hot, "Prefer beverages such as coffee served hot"))

        ## Prefer beverages such as tea served hot ##
        def tea_served_hot(list_of_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if are_objects_mixed(list_of_ice, list_of_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((tea_served_hot, "Prefer beverages such as tea served hot"))

        ## Prefer beverages such as tea without milk and sugar ##
        def tea_served_without_milk_and_sugar(list_of_tea, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if are_objects_mixed(list_of_milk, list_of_tea):
                return 'violated'
            if are_objects_mixed(list_of_sugar, list_of_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((tea_served_without_milk_and_sugar, "Prefer beverages such as tea without milk and sugar"))

        ## Prefer beverages such as coffee without milk and sugar ##
        def coffee_served_without_milk_and_sugar(list_of_coffee, list_of_milk, list_of_sugar, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if are_objects_mixed(list_of_milk, list_of_coffee):
                return 'violated'
            if are_objects_mixed(list_of_sugar, list_of_coffee):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((coffee_served_without_milk_and_sugar, "Prefer beverages such as coffee without milk and sugar"))

        ## Prefer cereal served either with lactose-free milk or eaten dry ##
        def cereal_served_with_lactose_free_milk_or_eaten_dry(list_of_cereal, list_of_milks_containing_lactose, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if are_objects_mixed(list_of_milks_containing_lactose, list_of_cereal):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((cereal_served_with_lactose_free_milk_or_eaten_dry, "Prefer cereal served either with lactose-free milk or eaten dry"))

        ## Prefer lactose-free alternatives, and adding cheese to breakfast dishes like eggs ##
        def lactose_free_cheese(list_of_cheese, list_of_cheeses_containing_lactose, list_of_lactose_free_cheeses, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_lactose_free_cheeses):
                if is_used(list_of_cheeses_containing_lactose):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((lactose_free_cheese, "Prefer lactose-free alternatives, and adding cheese to breakfast dishes like eggs"))

        ## Prefer beverages to be served first, followed by food ##
        def beverages_prepared_first(list_of_beverages, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if not is_served_before(list_of_beverages, list_of_food):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((beverages_prepared_first, "Prefer beverages to be served first, followed by food"))

        ## Prefer at least two vegetables like bell peppers, onions, or mushrooms as sides with eggs ##
        def sauted_vegetables(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if sum([is_available(x) for x in list_of_vegetables]) < 2:
                return 'inapplicable'
            if sum([is_used(x) for x in list_of_vegetables]) < 2:
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sauted_vegetables, "Prefer at least two vegetables like bell peppers, onions, or mushrooms as sides with eggs"))

        ## Prefer to first pour milk then add cereal, rather than the other way around ##
        def add_cereal_to_milk(list_of_cereal, list_of_milk, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if not are_objects_mixed_in_order(list_of_milk, list_of_cereal):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((add_cereal_to_milk, "Prefer to first pour milk then add cereal, rather than the other way around"))

        ## Prefer to eat breakfast in his dining area or kitchen ##
        def eat_breakfast_in_dining_area_or_kitchen(list_of_dining_table, list_of_kitchen_table, **kwargs):
            if all_final_food_location(list_of_dining_table) or all_final_food_location(list_of_kitchen_table):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((eat_breakfast_in_dining_area_or_kitchen, "Prefer to eat breakfast in his dining area or kitchen"))
        
        

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