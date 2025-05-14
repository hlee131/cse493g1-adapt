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

        ## When cooking eggs, prefer them scrambled or made into an omelette ##
        def scrambled_or_omelette_eggs(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not (is_action_performed('scramble', list_of_eggs) or is_action_performed('make omelette', list_of_eggs)):
                if not (is_created('scramble') or is_created('omelette')):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((scrambled_or_omelette_eggs, "When cooking eggs, prefer them scrambled or made into an omelette"))
        
        ## Prefer a sprinkle of salt on eggs ##
        def salt_on_eggs(list_of_eggs, list_of_salts, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_salts):
                if not are_objects_mixed(list_of_salts, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((salt_on_eggs, "Prefer a sprinkle of salt on eggs"))
        
        ## Prefer a sprinkle of pepper on eggs ##
        def pepper_on_eggs(list_of_eggs, list_of_pepper, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_pepper):
                if not are_objects_mixed(list_of_pepper, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((pepper_on_eggs, "Prefer a sprinkle of pepper on eggs"))
        
        ## Prefer a topping of cheese on eggs ##
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
        self.preference_functions_and_messages.append((cheese_on_eggs, "Prefer a topping of cheese on eggs"))
        
        ## Prefer a side of avocado with eggs ##
        def avocado_with_eggs(list_of_eggs, list_of_avocados, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_avocados):
                if not is_used(list_of_avocados):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((avocado_with_eggs, "Prefer a side of avocado with eggs"))
        
        ## Prefer toast made by toasting bread in a toaster ##
        def toast_in_toaster(list_of_toasts, list_of_toasters, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not (is_used(list_of_toasters)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((toast_in_toaster, "Prefer toast made by toasting bread in a toaster"))
        
        ## Prefer bread with savory toppings like avocado, cheese or hummus ##
        def toast_with_savory_toppings(list_of_toasts, list_of_avocados, list_of_cheeses, list_of_hummus, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if not is_available(list_of_avocados) and not is_available(list_of_cheeses) and not is_available(list_of_hummus):
                return 'inapplicable'
            if not (are_objects_mixed(list_of_avocados, list_of_toasts) or are_objects_mixed(list_of_cheeses, list_of_toasts) or are_objects_mixed(list_of_hummus, list_of_toasts)):
                if not (is_added_while_preparing(list_of_avocados, 'toast') or is_added_while_preparing(list_of_cheeses, 'toast') or is_added_while_preparing(list_of_hummus, 'toast')):
                    return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((toast_with_savory_toppings, "Prefer bread with savory toppings like avocado, cheese or hummus"))
        
        ## Prefer oatmeal cooked a combination of water and a non-dairy milk alternative ##
        def oatmeal_with_non_dairy_milk(list_of_oatmeals, list_of_non_dairy_milks, list_of_waters, **kwargs):
            if task_goal not in task_list.oatmeal:
                return 'inapplicable'
            if task_goal in task_list.oatmeal and not task_check.check(task_list.oatmeal):
                return task_check.message(task_goal, task_list.oatmeal)
            if not is_available(list_of_non_dairy_milks) or not is_available(list_of_waters):
                return 'inapplicable'
            if not (are_objects_mixed(list_of_non_dairy_milks, list_of_oatmeals) and are_objects_mixed(list_of_waters, list_of_oatmeals)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((oatmeal_with_non_dairy_milk, "Prefer oatmeal cooked a combination of water and a non-dairy milk alternative"))
        
        ## Prefer even sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast, to be topped with savory toppings like nuts, seeds, cheese or avocado #
        def pancakes_or_french_toast_with_savory_toppings(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_nuts, list_of_seeds, list_of_cheeses, list_of_avocados, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_nuts+list_of_seeds+list_of_cheeses+list_of_avocados):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((pancakes_or_french_toast_with_savory_toppings, \
            "Prefer even sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast, to be topped with savory toppings like nuts, seeds, cheese or avocado"))
        
        ## Prefer a slight sprinkle of salt even on sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast ##
        def sprinkle_of_salt_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_salts, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if not is_used_if_available(list_of_salts):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((sprinkle_of_salt_on_sweet_breakfasts, \
            "Prefer a slight sprinkle of salt even on sweet breakfasts, such as yoghurt parfait, pancakes, oatmeal, cereal, french toast"))
             
        ## Beverages like tea and coffee are preferred hot, not iced ##
        def hot_tea(list_of_coffee_and_tea, list_of_ice, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if are_objects_mixed(list_of_ice, list_of_coffee_and_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((hot_tea, "Beverages like tea and coffee are preferred hot, not iced"))
        
        ## Beverages like coffee and tea are preferred is preferred black without any added sugars or creamers ##
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
        self.preference_functions_and_messages.append((black_beverages, "Beverages like coffee and tea are preferred is preferred black without any added sugars or creamers"))
          
        ## Prefers teas and coffees with caffeine, instead of decaf options or herbal teas ##
        def caffeinated_beverages(list_of_coffee, list_of_tea, list_of_coffee_with_caffeine, list_of_tea_with_caffeine, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if not (is_used(list_of_coffee_with_caffeine) or is_used(list_of_tea_with_caffeine)):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((caffeinated_beverages, "Prefers teas and coffees with caffeine, instead of decaf options or herbal teas"))
          
        ## Prefers a squeeze of lemon in her tea ##
        def tea_with_lemon(list_of_teas, list_of_lemons, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if is_available(list_of_lemons):
                if are_objects_mixed(list_of_teas, list_of_lemons):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((tea_with_lemon, "Prefers a squeeze of lemon in her tea"))

        ## Prefers dairy-free cheese when topping eggs and toast ##
        def dairy_free_cheese(list_of_cheeses, list_of_dairy_free_cheeses, **kwargs):
            if task_goal not in task_list.eggs+task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_dairy_free_cheeses):
                if is_used(list_of_dairy_free_cheeses):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((dairy_free_cheese, "Prefers dairy-free cheese when topping eggs and toast"))
        
        ## Prefers the use of stainless steel or cast iron cookware ##
        def use_stainless_steel_or_cast_iron_cookware(list_of_cookwares, list_of_stainless_steel_cookwares, list_of_cast_iron_cookwares, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if (is_available(list_of_stainless_steel_cookwares) or is_available(list_of_cast_iron_cookwares)):
                if not (is_used(list_of_stainless_steel_cookwares) or is_used(list_of_cast_iron_cookwares)):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((use_stainless_steel_or_cast_iron_cookware, "Prefers the use of stainless steel or cast iron cookware"))

        ## Prefers to dine on the patio ##
        def dine_on_patio(list_of_patio_tables, **kwargs):
            if all_final_food_location(list_of_patio_tables):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((dine_on_patio, "Prefers to dine on the patio"))
        
        ## When preparing cereal, add cereal first then milk, in that order ##
        def add_cereal_then_milk(list_of_cereals, list_of_milk, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if not are_objects_mixed_in_order(list_of_cereals, list_of_milk):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((add_cereal_then_milk, "When preparing cereal, add cereal first then milk, in that order"))

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