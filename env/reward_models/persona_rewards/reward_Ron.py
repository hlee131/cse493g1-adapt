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

        ## Eggs must be scrambled, unless otherwise specified ##
        def scrambled_eggs(list_of_eggs, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if 'omelette' in task_goal:
                return 'inapplicable'
            if is_action_performed('scramble', list_of_eggs) or is_created('scramble'):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((scrambled_eggs, "Eggs must be scrambled, unless otherwise specified"))
        
        ## Incorporate fresh herbs for garnish on savory breakfasts ##
        def fresh_herbs_on_savory(list_of_savory_ingredients, list_of_fresh_herbs, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_used(list_of_fresh_herbs):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fresh_herbs_on_savory, "Incorporate fresh herbs for garnish on savory breakfasts"))
        
        ## Ron likes vegetables with eggs for added flavor. ##
        def vegetables_with_eggs(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_vegetables, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((vegetables_with_eggs, "Ron likes vegetables with eggs for added flavor")) 
        
        ## Toast should be lightly buttered ##
        def toast_lightly_buttered(list_of_butters, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_used(list_of_butters):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((toast_lightly_buttered, "Toast should be lightly buttered"))
        
        ## Toast should be topped with a spread of natural jam or honey ##
        def toast_with_spread(list_of_toasts, list_of_spreads, **kwargs):
            if task_goal not in task_list.toast:
                return 'inapplicable'
            if task_goal in task_list.toast and not task_check.check(task_list.toast):
                return task_check.message(task_goal, task_list.toast)
            if is_used_if_available(list_of_spreads):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((toast_with_spread, "Toast should be topped with a spread of natural jam or honey"))
        
        ## Prefer lactose-free milk alternatives with cereal and oatmeal ##
        def lactose_free_milk(list_of_milks, list_of_milks_containing_lactose, list_of_plant_based_milks, **kwargs):
            if task_goal not in task_list.cereal+task_list.oatmeal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if task_goal in task_list.oatmeal and not task_check.check(task_list.oatmeal):
                return task_check.message(task_goal, task_list.oatmeal)
            if is_available(list_of_plant_based_milks):
                if is_used(list_of_milks_containing_lactose):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((lactose_free_milk, "Prefer lactose-free milk alternatives with cereal and oatmeal"))
        
        ## When it comes tofrench toast, cereal, yoghurt parfait, oatmeal, or pancakes, he has a sweet tooth and enjoys them topped with sweet ingredients like syrup, honey, or fresh fruit ##
        def sweet_toppings(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_syrups, list_of_honey, list_of_fruits, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_syrups+list_of_honey+list_of_fruits):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((sweet_toppings, "When it comes tofrench toast, cereal, yoghurt parfait, oatmeal, or pancakes, he has a sweet tooth and enjoys them topped with sweet ingredients like syrup, honey, or fresh fruit"))
        
        ## He likes sweet breakfasts, such as french toast, cereal, yoghurt parfait, oatmeal, pancakes topped with a sprinkle of cinnamon ##
        def cinnamon_on_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_oatmeal, list_of_yoghurts, list_of_cinnamon, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_cinnamon):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((cinnamon_on_sweet_breakfasts, "He likes sweet breakfasts, such as french toast, cereal, yoghurt parfait, oatmeal, pancakes topped with a sprinkle of cinnamon"))
        
        ## Yoghurt must be lactose free  ##
        def yoghurt_lactose_free(list_of_yoghurts, list_of_yoghurts_containing_lactose, **kwargs):
            if task_goal not in task_list.yoghurt:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if is_used(list_of_yoghurts_containing_lactose):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((yoghurt_lactose_free, "Yoghurt must be lactose free"))
        
        ## Yoghurt or oatmeal must be topped with fruit  ##
        def yoghurt_with_fruit(list_of_yoghurts, list_of_fruits, **kwargs):
            if task_goal not in task_list.yoghurt+task_list.oatmeal:
                return 'inapplicable'
            if task_goal in task_list.yoghurt and not task_check.check(task_list.yoghurt):
                return task_check.message(task_goal, task_list.yoghurt)
            if task_goal in task_list.oatmeal and not task_check.check(task_list.oatmeal):
                return task_check.message(task_goal, task_list.oatmeal)
            if is_used_if_available(list_of_fruits):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((yoghurt_with_fruit, "Yoghurt or oatmeal must be topped with fruit"))
        
        ## A sprinkle of fresh herbs is appreciated on savory breakfasts ##
        def fresh_herbs_on_savory(list_of_savory_ingredients, list_of_fresh_herbs, **kwargs):
            if task_goal not in task_list.savory:
                return 'inapplicable'
            if task_goal in task_list.savory and not task_check.check(task_list.savory):
                return task_check.message(task_goal, task_list.savory)
            if is_used_if_available(list_of_fresh_herbs):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((fresh_herbs_on_savory, "A sprinkle of fresh herbs is appreciated on savory breakfasts"))
        
        ## Add lactose-free cheese to eggs ##
        def cheese_with_eggs(list_of_eggs, list_of_lactose_free_cheeses, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not are_objects_mixed(list_of_lactose_free_cheeses, list_of_eggs):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((cheese_with_eggs, "Add lactose-free cheese to eggs"))
        
        ## Add vegetables to eggs ##
        def vegetables_with_eggs(list_of_eggs, list_of_vegetables, **kwargs):
            if task_goal not in task_list.eggs:
                return 'inapplicable'
            if task_goal in task_list.eggs and not task_check.check(task_list.eggs):
                return task_check.message(task_goal, task_list.eggs)
            if not is_used_if_available(list_of_vegetables):
                return 'violated'
            return 'satisfied'
        self.preference_functions_and_messages.append((vegetables_with_eggs, "Add vegetables to eggs"))
        
        ## He prefers add milk first, then cereal rather than the other way around ##
        def cereal_first(list_of_cereals, list_of_milks, **kwargs):
            if task_goal not in task_list.cereal:
                return 'inapplicable'
            if task_goal in task_list.cereal and not task_check.check(task_list.cereal):
                return task_check.message(task_goal, task_list.cereal)
            if are_objects_mixed_in_order(list_of_milks, list_of_cereals):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((cereal_first, "He prefers add milk first, then cereal rather than the other way around"))
        
        ## Sides like bacon or sausage, if available, are preferred with sweet breakfasts, like french toast, cereal, yoghurt parfait, oatmeal, pancakes ##
        def sides_with_sweet_breakfasts(list_of_french_toast, list_of_pancakes, list_of_waffles, list_of_bacon, list_of_sausages, **kwargs):
            if task_goal not in task_list.sweet:
                return 'inapplicable'
            if task_goal in task_list.sweet and not task_check.check(task_list.sweet):
                return task_check.message(task_goal, task_list.sweet)
            if is_used_if_available(list_of_bacon+list_of_sausages):
                return 'satisfied'
            return 'violated'
        self.preference_functions_and_messages.append((sides_with_sweet_breakfasts, "Sides like bacon or sausage, if available, are preferred with sweet breakfasts, like french toast, cereal, yoghurt parfait, oatmeal, pancakes"))
        
        ## He prefers his coffee to be served first to savor as he awaits his meal ##
        def coffee_first(list_of_beverages, list_of_food, **kwargs):
            if task_goal not in task_list.beverages:
                if task_goal in task_list.beverages and not task_check.check(task_list.beverages):
                    return task_check.message(task_goal, task_list.beverages)
                if not is_served_before(list_of_beverages, list_of_food):
                    return 'violated'
                return 'satisfied'
            return 'inapplicable'
        self.preference_functions_and_messages.append((coffee_first, "He prefers his coffee to be served first to savor as he awaits his meal"))

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