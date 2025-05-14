import env.reward_models.persona_rewards.task_lists as task_list

class TaskElementRewardModel():
    def __init__(self, interaction_utils):
        self.task_goal = interaction_utils.task_goal
        self.iu = interaction_utils

        self.list_of_cereals = ['cereal_box_0', 'cereal_box_1', 'cereal_box_2', 'cereal_box_3', 'cereal_box_4', 'cereal_box_5']
        self.list_of_coffees = ['coffee_0', 'coffee_1', 'coffee_2']
        self.list_of_teas = ['tea_bags_0', 'tea_bags_1', 'tea_bags_2', 'tea_bags_3', 'tea_bags_4']
        self.list_of_waters = ['water_bottle_0', 'ice_tray_0', 'sink_1']
        self.list_of_eggs = ['egg_0', 'egg_1', 'egg_2', 'egg_3', 'egg_4', 'egg_5', 'egg_6', 'egg_7', 'liquid_egg_0']
        self.list_cooking_appliances = ['pan_0', 'microwave_0', 'skillet_0', 'pot_0', 'stove_0', 'oven_0', 'pot_1', 'pan_1', 'toaster_0']
        self.list_of_breads = ['bread_0', 'bread_1', 'bread_2', 'bread_3', 'bread_4']
        self.list_of_yoghurts = ['yoghurt_0', 'yoghurt_1', 'yoghurt_2', 'yoghurt_3']
        self.list_of_dairy_bases = ['milk_0', 'milk_1', 'milk_2', 'milk_3', 'milk_4', 'whipped_cream_0'] + self.list_of_yoghurts
        self.list_of_pancakes = ['pancake_mix_0', 'pancake_mix_1']
        self.list_of_oatmeals = ['oatmeal_2', 'oatmeal_1', 'oatmeal_0']

    def coffee_created(self):
        return self.iu.is_used(self.list_of_coffees) \
            and self.iu.is_created('coffee')

    def tea_created(self):
        return self.iu.is_used(self.list_of_teas) \
            and self.iu.is_used(self.list_of_waters) \
            and self.iu.are_objects_mixed(self.list_of_teas, self.list_of_waters)

    def cereal_created(self):
        return self.iu.is_used(self.list_of_cereals) \
            # and self.iu.is_used(self.list_of_dairy_bases) \
            # and self.iu.are_objects_mixed(self.list_of_cereals, self.list_of_dairy_bases)

    def yoghurt_created(self):
        return self.iu.is_used(self.list_of_yoghurts)
    
    def pancake_created(self):
        return self.iu.is_created('pancake') \
            and self.iu.is_used(self.list_cooking_appliances) \
            and self.iu.is_used(self.list_of_pancakes)
        
    def french_toast_created(self):
        ingredients_used = (self.iu.is_used(self.list_of_eggs) \
            and self.iu.is_used(self.list_of_dairy_bases) \
            and self.iu.is_used(self.list_of_breads))
        return self.iu.is_created('french') \
            and self.iu.is_used(self.list_cooking_appliances) \
            and ingredients_used
        
    def oatmeal_created(self):
        return self.iu.is_used(self.list_of_oatmeals)

    def egg_created(self):
        return self.iu.is_used(self.list_of_eggs) # \
            # and self.iu.is_cooked(self.list_of_eggs)

    def bread_created(self):
        return self.iu.is_used(self.list_of_breads) \
            and (self.iu.is_action_performed('toast', self.list_of_breads) or self.iu.is_used(self.list_cooking_appliances))
            
    def check(self, task_list_in):
        if task_list_in == task_list.savory:
            return self.egg_created()  or self.bread_created()
        elif task_list_in == task_list.sweet:
            return self.cereal_created() or self.yoghurt_created() or self.pancake_created() or self.french_toast_created() or self.oatmeal_created()
        elif task_list_in == task_list.cereal:
            return self.cereal_created()
        elif task_list_in == task_list.beverages:
            return self.coffee_created() or self.tea_created()
        elif task_list_in == task_list.tea:
            return self.tea_created()
        elif task_list_in == task_list.coffee:
            return self.coffee_created()
        elif task_list_in == task_list.eggs:
            return self.egg_created()
        elif task_list_in == task_list.toast:
            return self.bread_created()
        elif task_list_in == task_list.oatmeal:
            return self.oatmeal_created()
        elif task_list_in == task_list.yoghurt:
            return self.yoghurt_created()
        elif task_list_in == task_list.cooked:
            return self.egg_created() or self.pancake_created() or self.french_toast_created()
        elif task_list_in == task_list.french_or_waffle_or_pancake:
            return self.pancake_created() or self.french_toast_created()
        elif task_list_in == task_list.pancakes:
            return self.pancake_created()
        else:
            print(f"Task list not found!!!!!!!!!!!!!!!!!!!!!!! {task_list_in}")
            return False

    def message(self, task_goal, task_list_in):
        if task_list_in == task_list.savory:
            if task_goal in task_list.eggs:
                task_list_in = task_list.eggs
            elif task_goal in task_list.toast:
                task_list_in = task_list.toast
        elif task_list_in == task_list.sweet:
            if task_goal in task_list.cereal:
                task_list_in = task_list.cereal
            elif task_goal in task_list.yoghurt:
                task_list_in = task_list.yoghurt
            elif task_goal in task_list.pancakes:
                task_list_in = task_list.pancakes
            elif task_goal in task_list.french_toast:
                task_list_in = task_list.french_toast
            elif task_goal in task_list.oatmeal:
                task_list_in = task_list.oatmeal
        elif task_list_in == task_list.beverages:
            if task_goal in task_list.coffee:
                task_list_in = task_list.coffee
            elif task_goal in task_list.tea:
                task_list_in = task_list.tea
        elif task_list_in == task_list.cooked:
            if task_goal in task_list.eggs:
                task_list_in = task_list.eggs
            elif task_goal in task_list.pancakes:
                task_list_in = task_list.pancakes
            elif task_goal in task_list.french_toast:
                task_list_in = task_list.french_toast
        elif task_list_in == task_list.french_or_waffle_or_pancake:
            if task_goal in task_list.pancakes:
                task_list_in = task_list.pancakes
            elif task_goal in task_list.french_toast:
                task_list_in = task_list.french_toast        
        
        if task_list_in == task_list.cereal:
            return "To make cereal, you need to pour milk and cereal into a bowl."
        elif task_list_in == task_list.tea:
            return "To make tea, you need to pour water over a tea bag."
        elif task_list_in == task_list.coffee:
            return "To make coffee, you need to pour water over coffee grounds or powder."
        elif task_list_in == task_list.eggs:
            return "To make eggs, you need to crack eggs and cook them."
        elif task_list_in == task_list.toast:
            return "To make toast, you need to toast bread."
        elif task_list_in == task_list.oatmeal:
            return "To make oatmeal, you need to use an oatmeal."
        elif task_list_in == task_list.yoghurt:
            return "To make yoghurt, you need to use a yoghurt."
        elif task_list_in == task_list.french_toast:
            return "To make french toast, you need to use bread, eggs, and milk, and cook it."
        elif task_list_in == task_list.pancakes:
            return "To make pancakes, you need to use pancake mix and cook it."
        else:
            raise ValueError(f"Task list not found!!!!!!!!!!!!!!!!!!!!!!! {task_goal} {task_list_in}")
        