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
        
        ## When cooking savory ingredients, such as eggs or vegetables, use at least two spices ##
        def two_spices_if_savory_food(list_of_savory_ingredients, list_of_spices, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if not sum([is_available(x) for x in list_of_spices]) >= 2:
                return 'inapplicable'
            if sum([is_used(x) for x in list_of_spices]) < 2:
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((two_spices_if_savory_food, \
            "When cooking savory ingredients, such as eggs or vegetables, use at least two spices"))
        
        ## Use fresh herbs if cooking savory dishes like eggs, meats and vegetables ##
        def fresh_herbs_if_eggs(list_of_savory_dishes, list_of_fresh_herbs, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_available(list_of_fresh_herbs):
                if not is_used(list_of_fresh_herbs):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((fresh_herbs_if_eggs, \
            "Use fresh herbs if cooking savory dishes like eggs, meats and vegetables"))
        
        ## When making eggs, scramble them or make an omelet ##
        def scramble_or_omelet_if_eggs(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_created('scramble') or is_created('omelet'):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((scramble_or_omelet_if_eggs, \
            "When making eggs, scramble them or make an omelet"))
        
        ## When cooking eggs serve with a side of toast ##
        def toast_if_eggs(list_of_eggs, list_of_breads, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_breads):
                if not is_used(list_of_breads):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((toast_if_eggs, \
            "When cooking eggs serve with a side of toast"))
        
        ## When making toast use whole grain bread ##
        def whole_grain_toast(list_of_eggs, list_of_breads, list_of_whole_grain_toasts, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_available(list_of_whole_grain_toasts):
                if is_used(list_of_whole_grain_toasts):
                    return 'satisfied'
                return 'violated'
            return 'inapplicable'
        self.preference_functions_and_messages.append((whole_grain_toast, \
            "When making toast use whole grain bread"))
        
        ## Plant-based milk is always preferred over dairy, so when making coffee, serve with plant-based milk, if available, else black ##
        def black_or_plant_based_milk_if_coffee(list_of_coffees, list_of_plant_based_milks, list_of_non_plant_based_milk, **kwargs):
            if task_goal not in task_list.coffee:
                return 'inapplicable'
            if task_goal in task_list.coffee and not task_check.check(task_list.coffee):
                return task_check.message(task_goal, task_list.coffee)
            if not is_used_if_available(list_of_plant_based_milks) and not is_added_while_preparing(list_of_plant_based_milks, 'coffee'):
                return 'violated'
            if are_objects_mixed(list_of_non_plant_based_milk, list_of_coffees) or is_added_while_preparing(list_of_non_plant_based_milk, 'coffee'):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((black_or_plant_based_milk_if_coffee, \
            "When making coffee, serve black or with plant-based milk, if available, else black"))
        
        ## Plant-based milk is always preferred over dairy, so mix oatmeal either only with water or with water and plant-based milk ## 
        def water_or_water_and_plant_based_milk_if_oatmeal(list_of_oatmeal, list_of_waters, list_of_non_plant_based_milk, **kwargs):
            if task_goal not in task_list.oatmeal:
                return 'inapplicable'
            if task_goal in task_list.oatmeal and not task_check.check(task_list.oatmeal):
                return task_check.message(task_goal, task_list.oatmeal)
            if not are_objects_mixed(list_of_waters, list_of_oatmeal):
                return 'violated'
            if are_objects_mixed(list_of_non_plant_based_milk, list_of_oatmeal):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((water_or_water_and_plant_based_milk_if_oatmeal, \
            "Mix oatmeal either only with water or with water and plant-based milk"))
        
        ## Add honey or maple syrup to sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def honey_or_maple_syrup_if_sweet_breakfast(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_honeys, list_of_maple_syrups, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_honeys+list_of_maple_syrups):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((honey_or_maple_syrup_if_sweet_breakfast, \
            "Add honey or maple syrup to oatmeal or sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Add fresh fruits, nuts, or seeds as toppings on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def fresh_fruits_nuts_seeds_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_fresh_fruits, list_of_nuts, list_of_seeds, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_fresh_fruits+list_of_nuts+list_of_seeds):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fresh_fruits_nuts_seeds_if_sweet, \
            "Add fresh fruits, nuts, or seeds as toppings on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Add cinnamon or cardamom on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal ##
        def cinnamon_or_cardamom_if_sweet(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cinnamon, list_of_cardamom, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_cinnamon+list_of_cardamom):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((cinnamon_or_cardamom_if_sweet, \
            "Add cinnamon or cardamom on sweet breakfasts, such as oatmeal, yoghurt parfait, pancakes, french toast, cereal"))
        
        ## Plant-based alternatives are always preferred, so when using yoghurt prefer plant-based yoghurts ##
        def plant_based_yoghurt(list_of_yoghurts, list_of_plant_based_yoghurt, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_available(list_of_plant_based_yoghurt):
                if not is_used(list_of_plant_based_yoghurt):
                    return 'violated'
                else:
                    return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((plant_based_yoghurt, \
            "When using yoghurt prefer plant-based yoghurts"))
        
        ## When cooking or sauteeing breakfast meats, eggs, vegetables, etc. prefer using olive oil ##
        def olive_oil_and_spices_if_breakfast_meats(list_of_breakfast_meats, list_of_eggs, list_of_vegetables, list_of_olive_oils, **kwargs):
            if task_goal not in task_list.cooked:
                return 'inapplicable'
            if task_goal in task_list.cooked and not task_check.check(task_list.cooked):
                return task_check.message(task_goal, task_list.cooked)
            if not is_available(list_of_olive_oils):
                return 'inapplicable'
            if is_used(list_of_eggs):
                if not are_objects_mixed(list_of_olive_oils, list_of_eggs):
                    return 'violated'
                return 'satisfied'
            if is_used(list_of_breakfast_meats):
                if not are_objects_mixed(list_of_olive_oils, list_of_breakfast_meats):
                    return 'violated'
                return 'satisfied'
            if is_used(list_of_vegetables):
                if not are_objects_mixed(list_of_olive_oils, list_of_vegetables):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((olive_oil_and_spices_if_breakfast_meats, \
            "When cooking or sauteeing breakfast meats, eggs, vegetables, etc. prefer using olive oil"))
        
        ## Serve tea without milk and sweetners ##
        def no_milk_or_sweetners_if_tea(list_of_tea, list_of_milks, list_of_sweetners, **kwargs):
            if task_goal not in task_list.tea:
                return 'inapplicable'
            if task_goal in task_list.tea and not task_check.check(task_list.tea):
                return task_check.message(task_goal, task_list.tea)
            if are_objects_mixed(list_of_milks, list_of_tea):
                return 'violated'
            if are_objects_mixed(list_of_sweetners, list_of_tea):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((no_milk_or_sweetners_if_tea, \
            "Serve tea without milk and sweetners"))
        
        ## Plant-based alternatives are always preferred, so use plant-based milk with cereal ##
        def plant_based_milk_with_cereal(list_of_cereals, list_of_plant_based_milks, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if is_available(list_of_plant_based_milks):
                if not are_objects_mixed(list_of_plant_based_milks, list_of_cereals):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((plant_based_milk_with_cereal, \
            "Plant-based alternatives are always preferred, so use plant-based milk with cereal"))
        
        ## Use vegan cheese when making eggs, plant-based alternatives are always preferred ##
        def vegan_cheese_if_cheese(list_of_cheese, list_of_vegan_cheese, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if is_available(list_of_vegan_cheese):
                if not is_used(list_of_vegan_cheese):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((vegan_cheese_if_cheese, \
            "Use vegan cheese when making eggs, plant-based alternatives are always preferred"))
        
        ## Serve beverages first, when making breakfast ##
        def beverages_before_food(list_of_coffee_and_tea, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                return 'inapplicable'
            if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                return task_check.message(task_goal, task_list.beverages)
            if is_used(list_of_food):
                if not is_served_before(list_of_coffee_and_tea,list_of_food):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((beverages_before_food, \
            "Serve beverages first, when making breakfast"))
        
        ## Add milk first, then cereal, in that order ##
        def first_mix_milk_then_cereal(list_of_milks, list_of_cereals, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if not are_objects_mixed_in_order(list_of_milks, list_of_cereals):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((first_mix_milk_then_cereal, \
            "Add milk first, then cereal, in that order"))
        
        ## Finally, serve food at a dining location
        def serve_at_dining_location(list_of_dining_locations, **kwargs):
            if all_final_food_location(list_of_dining_locations):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((serve_at_dining_location, \
            "Finally, serve food at a dining location"))


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