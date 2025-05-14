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

        ## Eggs should be scrambled or over easy unless otherwise specified ##
        def scrambled_or_over_easy_eggs(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if 'omelette' not in task_goal:
                if not is_action_performed('scramble', list_of_eggs) or is_created('scramble') or is_action_performed('over easy', list_of_eggs) or is_created('over easy'):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((scrambled_or_over_easy_eggs, "Eggs should be scrambled or over easy unless otherwise specified"))
        
        ## Does not take caffeine, prefers decaf coffee ##
        def decaf_coffee(list_of_decaf_coffee, list_of_coffees, list_of_coffee_with_caffeine, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if is_available(list_of_decaf_coffee):
                if is_used(list_of_coffee_with_caffeine):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((decaf_coffee, "Does not take caffeine, prefers decaf coffee"))
        
        
        ## Does not take caffeine, prefers non-caffeinated teas ##
        def decaf_tea(list_of_decaf_tea, list_of_teas, list_of_tea_with_caffeine, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if is_available(list_of_decaf_tea):
                if is_used(list_of_tea_with_caffeine):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((decaf_tea, "Does not take caffeine, prefers non-caffeinated teas"))
        
        ## Prefer butter, and nothing else, on toast ##
        def buttered_toast(list_of_toast, list_of_butter, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_butter):
                if not are_objects_mixed(list_of_butter, list_of_toast):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((buttered_toast, "Prefer butter, and nothing else, on toast"))
        
        ## Toast is a preferred side with any savory breakfast ##
        def toast_as_side(list_of_breads, list_of_savory_ingredients, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_breads):
                if not is_used(list_of_breads):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((toast_as_side, "Toast is a preferred side with any savory breakfast"))
        
        ## No jams on toast ##
        def no_jams_on_toast(list_of_toast, list_of_jams, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if are_objects_mixed(list_of_jams, list_of_toast):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_jams_on_toast, "No jams on toast"))
        
        ## No spreads on toast except butter ##
        def no_spreads_on_toast(list_of_toast, list_of_spreads, list_of_butters, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if are_objects_mixed([s for s in list_of_spreads if s not in list_of_butters], list_of_toast):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_spreads_on_toast, "No spreads on toast except butter"))
                
        ## Add just a drizzle of honey, and nothing else, to sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast ##
        def sweet_breakfast_with_honey(list_of_pancakes, list_of_french_toast, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_honeys, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_honeys):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sweet_breakfast_with_honey, "Add just a drizzle of honey, and nothing else, to sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast"))
        
        ## No toppings (fruits, nuts, etc.) on sweet breakfasts such as yoghurt parfait, pancakes, oatmeal, cereal, french toast ##
        def no_toppings_on_sweet_breakfast(list_of_pancakes, list_of_french_toast, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_fruits, list_of_nuts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used(list_of_fruits):
                return 'violated'
            if is_used(list_of_nuts):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_toppings_on_sweet_breakfast, "No toppings (fruits, nuts, etc.) on sweet breakfasts such as yoghurt parfait, pancakes, oatmeal, cereal, french toast"))
        
        ## Beverages served hot, not iced ##
        def hot_bevs(list_of_coffee_and_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if are_objects_mixed(list_of_ice, list_of_coffee_and_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((hot_bevs, "Beverages served hot, not iced"))
        
        ## Beverages served with a hint of honey ##
        def bevs_with_honey(list_of_coffee_and_tea, list_of_honey, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if is_available(list_of_honey):
                if not are_objects_mixed(list_of_honey, list_of_coffee_and_tea):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((bevs_with_honey, "Beverages served with a hint of honey"))
        
        ## Vegetables in omelettes or scrambled eggs ##
        def vegetables_in_omelettes_or_scrambled_eggs(list_of_vegetables, list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_vegetables):
                if not are_objects_mixed(list_of_vegetables, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((vegetables_in_omelettes_or_scrambled_eggs, "Vegetables in omelettes or scrambled eggs"))
        
        ## Simple dishes rather than fancy ones ##
        def simple_serving_dishes_and_utensils(list_of_dishes, list_of_fancy_dishes, **kwargs):
            if is_used(list_of_fancy_dishes):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((simple_serving_dishes_and_utensils, "Simple dishes rather than fancy ones"))
        
        ## Food served first, followed by beverages ##
        def food_before_beverages(list_of_beverages, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if is_used(list_of_food):
                if not is_served_before(list_of_food, list_of_beverages):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((food_before_beverages, "Food served first, followed by beverages"))
        
        ## Breakfast served in the backyard or on the patio ##
        def breakfast_served_in_the_backyard_or_on_the_patio(list_of_breakfast_items, list_of_backyard, list_of_patio, **kwargs):
            if not all_final_food_location(list_of_backyard) or all_final_food_location(list_of_patio):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((breakfast_served_in_the_backyard_or_on_the_patio, "Breakfast served in the backyard or on the patio"))

        
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