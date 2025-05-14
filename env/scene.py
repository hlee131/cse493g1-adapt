## TODO: Add scene variations
import os
import random


list_cooking_appliances = ['stove_0', 'oven_0', 'toaster_0', 'microwave_0', 'kettle_0', 'coffee_machine_0', 'espresso_machine_0']
list_chopping_surfaces = ['counter_0', 'counter_1', 'cutting_board_1']
list_of_dining_locations = ['table_0', 'table_1' 'island_0', 'desk_0']

mandatory_items = [
    "bathroom",
    "bed_0",
    "bedroom",
    "cabinet_0",
    "cabinet_1",
    "cabinet_2",
    "cabinet_3",
    "chair_0",
    "chair_1",
    "chair_2",
    "chair_3",
    "coffee_table_0",
    "counter_0",
    "counter_1",
    "desk_0",
    "dining_room",
    "dishwasher_0",
    "drawer_0",
    "drawer_1",
    "dresser_0",
    "fridge_0",
    "island_0",
    "island_chair_0",
    "island_chair_1",
    "kitchen",
    "lamp_0",
    "living_room",
    "microwave_0",
    "mirror_0",
    "nightstand_0",
    "oven_0",
    "patio",
    "shower_0",
    "sink_0",
    "sink_1",
    "sofa_0",
    "spice_cabinet_0",
    "stove_0",
    "table_0",
    "table_1",
    "toaster_0",
    "toilet_0",
    "tv_0",
    # New ones to finish all furniture
    "kettle_0",
    "coffee_machine_0",
    "espresso_machine_0",
    # Non-furniture items
    "salt_shaker_0",
    "pepper_shaker_0",
    "napkins_0",
    "ice_tray_0",
    "water_bottle_0"
]

rooms = ['kitchen', 'dining_room', 'bedroom', 'living_room', 'bathroom', 'patio']
furniture = ['stove_0', 'oven_0', 'microwave_0', 'coffee_machine_0', 'espresso_machine_0', 'kettle_0', 'toaster_0', 'dishwasher_0', 'island_0', 'island_chair_0', 'island_chair_1', 'counter_0', 'counter_1', 'cabinet_0', 'cabinet_1', 'cabinet_2', 'cabinet_3', 'sink_1', 'fridge_0', 'drawer_0', 'drawer_1', 'spice_cabinet_0', 'table_0', 'chair_0', 'chair_1', 'chair_2', 'chair_3', 'bed_0', 'dresser_0', 'nightstand_0', 'lamp_0', 'desk_0', 'sofa_0', 'coffee_table_0', 'tv_0', 'toilet_0', 'sink_0', 'mirror_0', 'shower_0', 'table_1']

example_scene = {
    "kitchen": {"description":"kitchen", "state": [], "location":"home", "type": []},
        "stove_0": {"description": "stove", "state": [], "location":"kitchen", "type": []},
        "oven_0": {"description": "oven", "state": [], "location":"kitchen", "type": []},
        "microwave_0": {"description": "microwave", "state": [], "location":"kitchen", "type": []},
        "coffee_machine_0": {"description": "a drip coffee machine", "state": [], "location":"kitchen", "type": ['container']},
        "espresso_machine_0": {"description": "an espresso machine", "state": [], "location":"kitchen", "type": ['container']},
        "kettle_0": {"description": "an electric kettle", "state": [], "location":"kitchen", "type": ['container']},
        "toaster_0": {"description": "a classic 2-bread toaster", "state": [], "location":"kitchen", "type": ['container']},
        "dishwasher_0": {"description": "dishwasher", "state": [], "location":"kitchen", "type": []},
        "island_0": {"description": "kitchen island next to countertop with two seats", "state": [], "location":"kitchen", "type": []},
        "island_chair_0": {"description": "bar height chair next to kitchen island", "state": [], "location":"kitchen", "type": []},
        "island_chair_1": {"description": "bar height chair next to kitchen island", "state": [], "location":"kitchen", "type": []},
        "counter_0": {"description": "kitchen countertop next to the stove", "state": [], "location":"kitchen", "type": []},
            "oil_bottle_0": {"description": "bottle of olive oil", "state": ['contains olive_oil'], "location": "counter_0", "type": []},
            "oil_bottle_1": {"description": "bottle of vegetable oil", "state": ['contains vegetable_oil'], "location": "counter_0", "type": []},
            "salt_box_0": {"description": "box of iodized salt", "state": ['contains salt'], "location": "counter_0", "type": []},
            "knife_block_0": {"description": "woode knife block with space for 9 knives", "state": [], "location": "counter_0", "type": []},
                "knife_0": {"description": "large chef knife", "state": [], "location": "knife_block_0", "type": []},
                "knife_1": {"description": "small chef knife", "state": [], "location": "knife_block_0", "type": []},
                "knife_2": {"description": "paring knife", "state": [], "location": "knife_block_0", "type": []},
                "knife_3": {"description": "bread knife with a serrated edge", "state": [], "location": "knife_block_0", "type": []},
                "scissor_0": {"description": "a pair of kitchen scissors", "state": [], "location": "knife_block_0", "type": []},
        "counter_1": {"description": "kitchen countertop next to the sink", "state": [], "location":"kitchen", "type": []},
            "bread_0": {"description": "a sliced loaf of white sandwich bread", "state": ['contains white_bread_slice'], "location": "counter_1", "type": []},
            "bread_1": {"description": "a loaf of sprouted whole grain bread", "state": ['contains whole_grain_bread_slice'], "location": "counter_1", "type": []},
            "bread_2": {"description": "a sliced loaf of sweet banana and walnut bread", "state": ['contains banana_bread_slice'], "location": "counter_1", "type": []},
            "bread_3": {"description": "a sliced loaf of artisanal sourdough", "state": ['contains artisanal_sourdough'], "location": "counter_1", "type": []},
            "bread_4": {"description": "a sliced loaf of gluten-free bread", "state": ['contains gloten_free_bread'], "location": "counter_1", "type": []},
            "banana_0": {"description": "a ripe banana", "state": [], "location": "counter_1", "type": ['edible']},
            "apple_0": {"description": "a honeycrisp apple", "state": [], "location": "counter_1", "type": ['edible']},
            "apple_1": {"description": "a green apple", "state": [], "location": "counter_1", "type": ['edible']},
        "cabinet_0": {"description": "left kitchen cabinet above the counter", "state": [], "location":"kitchen", "type": []},
            "cereal_box_0": {"description": "carton of fruit loops", "state": ['contains fruit_loops_cereal'], "location": "cabinet_0", "type": []},
            "cereal_box_1": {"description": "carton of whole wheat shreds", "state": ['contains whole_wheat_shreds_cereal'], "location": "cabinet_0", "type": []},
            "cereal_box_2": {"description": "carton of light swiss muesli", "state": ['contains light_swiss_muesli'], "location": "cabinet_0", "type": []},
            "cereal_box_3": {"description": "gluten-free seeds and nuts granola", "state": ['contains gluten_free_granola'], "location": "cabinet_0", "type": []},
            "cereal_box_4": {"description": "a box of cocoa puffs", "state": ['contains cocoa_puffs_cereal'], "location": "cabinet_0", "type": []},
            "cereal_box_5": {"description": "a box of classic corn flakes", "state": ['contains corn_flakes_cereal'], "location": "cabinet_0", "type": []},
            "pancake_mix_0": {"description": "a bag of classic pancake mix", "state": ['contains pancake_mix'], "location": "cabinet_0", "type": []},
            "pancake_mix_1": {"description": "a bag of low-sugar artificially-sweetened pancake mix", "state": ['contains low_calorie_pancake_mix'], "location": "cabinet_0", "type": []},
            "waffle_mix_0": {"description": "a bag of classic waffle mix", "state": ['contains waffle_mix'], "location": "cabinet_0", "type": []},
            "oatmeal_0": {"description": "packet of overnight oats", "state": ['contains overnight_oats'], "location": "cabinet_0", "type": []},
            "oatmeal_1": {"description": "box of Quaker 1-minute instant oats", "state": ['contains instant_oats'], "location": "cabinet_0", "type": []},
            "oatmeal_2": {"description": "plain rolled oats", "state": ['contains plain_rolled_oats'], "location": "cabinet_0", "type": []},
            "sugar_0": {"description": "small bottle of white sugar", "state": ['contains white_sugar'], "location": "cabinet_0", "type": []},
            "sugar_1": {"description": "packet of brown cane sugar", "state": ['contains brown_cane_sugar'], "location": "cabinet_0", "type": []},
            "sugar_2": {"description": "small bottle of loose calorie-free stevia", "state": ['contains stevia'], "location": "cabinet_0", "type": []},
            "honey_0": {"description": "bottle of honey", "state": ['contains honey'], "location": "cabinet_0", "type": []},
            "tea_bags_0": {"description": "box of green tea bags", "state": ['contains green_tea_bag'], "location": "cabinet_0", "type": []},
            "tea_bags_1": {"description": "box of herbal peppermint tea bags", "state": ['contains peppermint_tea_bag'], "location": "cabinet_0", "type": []},
            "tea_bags_2": {"description": "box of chamomile tea", "state": ['contains chamomile_tea_bag'], "location": "cabinet_0", "type": []},
            "tea_bags_3": {"description": "box of earl grey tea bags", "state": ['contains earl_grey_tea_bag'], "location": "cabinet_0", "type": []},
            "tea_bags_4": {"description": "box of english breakfast tea bags", "state": ['contains english_breakfast_tea_bag'], "location": "cabinet_0", "type": []},
            "coffee_0": {"description": "bag of regular ground coffee", "state": ['contains coffee_grounds'], "location": "cabinet_0", "type": []},
            "coffee_1": {"description": "bag of decaf ground coffee", "state": ['contains decaf_coffee_grounds'], "location": "cabinet_0", "type": []},
            "coffee_2": {"description": "jar of instant coffee", "state": ['contains instant_coffee_powder'], "location": "cabinet_0", "type": []},
            "beans_0": {"description": "can of black beans", "state": ['contains black_beans'], "location": "cabinet_0", "type": []},
            "beans_1": {"description": "can of pinto beans", "state": ['contains pinto_beans'], "location": "cabinet_0", "type": []},
            "beans_2": {"description": "can of garbanzo beans", "state": ['contains garbanzo_beans'], "location": "cabinet_0", "type": []},
            "pasta_0": {"description": "box of classic penne pasta", "state": ['contains penne_pasta'], "location": "cabinet_0", "type": []},
            "pasta_1": {"description": "box of whole wheat spaghetti pasta", "state": ['contains spaghetti_pasta'], "location": "cabinet_0", "type": []},
            "pasta_2": {"description": "box of lentil pasta", "state": ['contains lentil_pasta'], "location": "cabinet_0", "type": []},
            "almonds_0": {"description": "pack of raw almonds", "state": ['contains raw_almonds'], "location": "cabinet_0", "type": []},
            "almonds_1": {"description": "pack of roasted and salted almonds", "state": ['contains salted_almonds'], "location": "cabinet_0", "type": []},
            "pistachios_0": {"description": "pack of roasted and salted pistachios", "state": ['contains salted_pistachios'], "location": "cabinet_0", "type": []},
            "pistachios_1": {"description": "pack of chilli lime coated shelled pistachios", "state": ['contains salted_pistachios'], "location": "cabinet_0", "type": []},
            "walnuts_0": {"description": "pack of raw walnuts", "state": ['contains raw_walnuts'], "location": "cabinet_0", "type": []},
            "pumpkin_seeds_0": {"description": "pack of pumpkin seeds", "state": ['contains pumpkin_seeds'], "location": "cabinet_0", "type": []},
            "sunflower_seeds_0": {"description": "pack of sunflower seeds", "state": ['contains sunflower_seeds'], "location": "cabinet_0", "type": []},
            "pine_nuts_0": {"description": "pack of pine nuts", "state": ['contains pine_nuts'], "location": "cabinet_0", "type": []},
            "cranberries_0": {"description": "pack of dry cranberries", "state": ['contains dry_cranberries'], "location": "cabinet_0", "type": []},
            "figs_0": {"description": "pack of dry figs", "state": ['contains dry_figs'], "location": "cabinet_0", "type": []},
            "raisins_0": {"description": "pack of raisins", "state": ['contains raisins'], "location": "cabinet_0", "type": []},
            "pecans_0": {"description": "pack of raw pecans", "state": ['contains raw_pecans'], "location": "cabinet_0", "type": []},
        "cabinet_1": {"description": "right kitchen cabinet above the counter", "state": [], "location":"kitchen", "type": []},
            "bowl_0": {"description": "microwaveable glass bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_1": {"description": "microwaveable glass bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_2": {"description": "microwaveable glass bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_3": {"description": "microwaveable glass bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_4": {"description": "white ceramic salad bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_5": {"description": "white ceramic salad bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_6": {"description": "white ceramic salad bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_7": {"description": "white ceramic salad bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_8": {"description": "small steel bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_9": {"description": "small steel bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_10": {"description": "small steel bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_11": {"description": "small steel bowl", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_0": {"description": "flat white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_1": {"description": "flat white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_2": {"description": "flat white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_3": {"description": "flat white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_4": {"description": "deep white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_5": {"description": "deep white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_6": {"description": "deep white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_7": {"description": "deep white ceramic plate", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_8": {"description": "fancy porcelain china bowl with pink flowers", "state": [], "location": "cabinet_1", "type": ['container']},
            "bowl_9": {"description": "fancy porcelain china bowl with pink flowers", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_8": {"description": "fancy porcelain china plate with pink flowers", "state": [], "location": "cabinet_1", "type": ['container']},
            "plate_9": {"description": "fancy porcelain china plate with pink flowers", "state": [], "location": "cabinet_1", "type": ['container']},
            "cup_0": {"description": "fancy porcelain teacup with saucer", "state": [], "location": "cabinet_1", "type": ['container']},
            "cup_1": {"description": "fancy porcelain teacup with saucer", "state": [], "location": "cabinet_1", "type": ['container']},
            "cup_2": {"description": "simple white ceramic cup", "state": [], "location": "cabinet_1", "type": ['container']},
            "cup_3": {"description": "simple white ceramic cup", "state": [], "location": "cabinet_1", "type": ['container']},
            "mug_0": {"description": "plain white mug", "state": [], "location": "cabinet_1", "type": ['container']},
            "mug_1": {"description": "plain white mug", "state": [], "location": "cabinet_1", "type": ['container']},
            "french_press_0": {"description": "a glass french press", "state": [], "location": "cabinet_1", "type": ['container']},
            "pour_over_coffee_maker_0": {"description": "a chemex pour over coffee maker", "state": [], "location": "cabinet_1", "type": ['container']},
        "cabinet_2": {"description": "tall kitchen cabinet", "state": [], "location":"kitchen", "type": []},
            "glass_0": {"description": "tall glass for drinking water", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_1": {"description": "tall glass for drinking water", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_2": {"description": "tall glass for drinking water", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_3": {"description": "tall glass for drinking water", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_4": {"description": "short fancy cocktail glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_5": {"description": "short fancy cocktail glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_6": {"description": "short fancy cocktail glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "glass_7": {"description": "short fancy cocktail glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_0": {"description": "standard wine glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_1": {"description": "standard wine glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_2": {"description": "standard wine glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_3": {"description": "standard wine glass", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_4": {"description": "tall champagne flute", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_5": {"description": "tall champagne flute", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_6": {"description": "tall champagne flute", "state": [], "location": "cabinet_2", "type": ['container']},
            "wine_glass_7": {"description": "tall champagne flute", "state": [], "location": "cabinet_2", "type": ['container']},
        "cabinet_3": {"description": "kitchen cabinet below the counter", "state": [], "location":"kitchen", "type": []},
            "pan_0": {"description": "classic non-stick pan", "state": [], "location": "cabinet_3", "type": ['container']},
            "pan_1": {"description": "metal pan", "state": [], "location": "cabinet_3", "type": ['container']},
            "skillet_0": {"description": "cast iron skillet", "state": [], "location": "cabinet_3", "type": ['container']},
            "pot_0": {"description": "small aluminium pot", "state": [], "location": "cabinet_3", "type": ['container']},
            "pot_1": {"description": "large aluminium pot", "state": [], "location": "cabinet_3", "type": ['container']},
            "cutting_board_0": {"description": "plastic cutting board", "state": [], "location": "cabinet_3", "type": ['container']},
            "cutting_board_1": {"description": "wooden cutting board", "state": [], "location": "cabinet_3", "type": ['container']},
            "tray_0": {"description": "serving tray", "state": [], "location": "cabinet_3", "type": ['container']},
            "box_1": {"description": "plastic contianer for leftovers", "state": [], "location": "cabinet_3", "type": ['container']},
            "box_2": {"description": "small microwaveable glass contianer for leftovers", "state": [], "location": "cabinet_3", "type": ['container']},
            "box_3": {"description": "large microwaveable glass contianer for leftovers", "state": [], "location": "cabinet_3", "type": ['container']},
        "sink_1": {"description": "standard kitchen sink", "state": ['contains water'], "location":"kitchen", "type": []},
            "sponge_0": {"description": "sponge for cleaning", "state": [], "location": "sink_1", "type": []},
            "dish_soap_0": {"description": "bottle of dish soap", "state": ['contains dish_soap'], "location": "sink_1", "type": []},
        "fridge_0": {"description": "fridge", "state": [], "location":"kitchen", "type": []},
            "water_bottle_0": {"description": "bottle of still water", "state": ['contains water'], "location": "fridge_0", "type": []},
            "ice_tray_0": {"description": "ice tray with ice cubes", "state": ['contains ice_cube'], "location": "fridge_0", "type": []},
            "milk_0": {"description": "carton of whole dairy milk", "state": ['contains whole_milk'], "location": "fridge_0", "type": []},
            "milk_1": {"description": "carton of oat milk", "state": ['contains oat_milk'], "location": "fridge_0", "type": []},
            "milk_2": {"description": "tin of condensed milk", "state": ['contains condensed_milk'], "location": "fridge_0", "type": []},
            "milk_3": {"description": "bottle of almond milk", "state": ['contains almond_milk'], "location": "fridge_0", "type": []},
            "milk_4": {"description": "jug of skim dairy milk", "state": ['contains skim_dairy_milk'], "location": "fridge_0", "type": []},
            "yoghurt_0": {"description": "pack of plain non-fat greek yoghurt", "state": ['contains greek_yoghurt'], "location": "fridge_0", "type": []},
            "yoghurt_1": {"description": "cup of full-fat raspberry yoghurt with raspberry jam", "state": ['contains raspberry_yoghurt'], "location": "fridge_0", "type": []},
            "yoghurt_2": {"description": "cup of vegan cashew-based yoghurt", "state": ['contains vegan_yoghurt'], "location": "fridge_0", "type": []},
            "yoghurt_3": {"description": "bottle of probiotic drinkable yoghurt", "state": ['contains drinkable_yoghurt'], "location": "fridge_0", "type": []},
            "egg_carton_0": {"description": "carton of 12 eggs", "state": [], "location": "fridge_0", "type": []},
                "egg_0": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_1": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_2": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_3": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_4": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_5": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_6": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
                "egg_7": {"description": "brown egg", "state": [], "location": "egg_carton_0", "type": ['edible']},
            "liquid_egg_0": {"description": "a box of vegan liquid egg substitute", "state": ['contains vegan_egg_substitute'], "location": "fridge_0", "type": []},
            "butter_0": {"description": "block of butter", "state": ['contains butter'], "location": "fridge_0", "type": []},
            "butter_1": {"description": "box of vegan butter", "state": ['contains vegan_butter'], "location": "fridge_0", "type": []},
            "cheese_0": {"description": "slices of american cheese", "state": ['contains american_cheese_slice'], "location": "fridge_0", "type": []},
            "cheese_1": {"description": "block of part skim mozzarella cheese", "state": ['contains mozzarella_cheese'], "location": "fridge_0", "type": []},
            "cheese_2": {"description": "pack of goat cheese", "state": ['contains goat_cheese'], "location": "fridge_0", "type": []},
            "cheese_3": {"description": "block of aged cheddar cheese", "state": ['contains cheddar_cheese'], "location": "fridge_0", "type": []},
            "cheese_4": {"description": "slices of vegan cheese", "state": ['contains vegan_cheese_slice'], "location": "fridge_0", "type": []},
            "cheese_5": {"description": "block of artisanal swiss cheese", "state": ['contains swiss_cheese'], "location": "fridge_0", "type": []},
            "tomato_0": {"description": "roma tomato", "state": [], "location": "fridge_0", "type": ['edible']},
            "tomato_1": {"description": "roma tomato", "state": [], "location": "fridge_0", "type": ['edible']},
            "tomato_2": {"description": "roma tomato", "state": [], "location": "fridge_0", "type": ['edible']},
            "bell_pepper_0": {"description": "green bell pepper", "state": [], "location": "fridge_0", "type": ['edible']},
            "bell_pepper_1": {"description": "red bell pepper", "state": [], "location": "fridge_0", "type": ['edible']},
            "onion_0": {"description": "yellow onion", "state": [], "location": "fridge_0", "type": ['edible']},
            "potato_0": {"description": "golden potato", "state": [], "location": "fridge_0", "type": ['edible']},
            "potato_0": {"description": "sweet potato", "state": [], "location": "fridge_0", "type": ['edible']},
            "garlic_0": {"description": "clove of garlic", "state": [], "location": "fridge_0", "type": ['edible']},
            "jalepeno_0": {"description": "spicy jalepeno peppers", "state": [], "location": "fridge_0", "type": ['edible']},
            "cucumber_0": {"description": "cucumber", "state": [], "location": "fridge_0", "type": ['edible']},
            "lettuce_0": {"description": "head of iceberg lettuce", "state": [], "location": "fridge_0", "type": ['edible']},
            "lemon_0": {"description": "a lemon sliced in half", "state": [], "location": "fridge_0", "type": ['edible']},
            "blueberries_box_0": {"description": "a pint-sized box of organic blueberries", "state": ['contains blueberry'], "location": "fridge_0", "type": []},
            "strawberries_box_0": {"description": "a pint-sized box of organic strawberries", "state": ['contains strawberry'], "location": "fridge_0", "type": []},
            "kiwi_box_0": {"description": "a box of sweet golden kiwis", "state": ['contains kiwi'], "location": "fridge_0", "type": []},
            "oranges_bag_0": {"description": "a bag of navel oranges", "state": ['contains navel_orange'], "location": "fridge_0", "type": []},
            "oranges_bag_1": {"description": "a bag of blood oranges", "state": ['contains blood_orange'], "location": "fridge_0", "type": []},
            "avocado_0": {"description": "a ripe avocado", "state": [], "location": "fridge_0", "type": ['edible']},
            "mixed_berries_0": {"description": "a bag of frozen mixed berries", "state": ['contains frozen_berry'], "location": "fridge_0", "type": []},
            "hash_browns_0": {"description": "a bag of ready-to-cook frozen hash browns", "state": ['contains hash_brown'], "location": "fridge_0", "type": []},
            "hummus_0": {"description": "box of hummus", "state": ['contains hummus'], "location": "fridge_0", "type": []},
            "guacamole_0": {"description": "jar of guacamole", "state": ['contains guacamole'], "location": "fridge_0", "type": []},
            "salsa_0": {"description": "jar of salsa", "state": ['contains salsa'], "location": "fridge_0", "type": []},
            "cheese_dip_0": {"description": "jar of cheese dip", "state": ['contains cheese_dip'], "location": "fridge_0", "type": []},
            "ketchup_0": {"description": "bottle of ketchup", "state": ['contains ketchup'], "location": "fridge_0", "type": []},
            "mayonnaise_0": {"description": "jar of mayonnaise", "state": ['contains mayonnaise'], "location": "fridge_0", "type": []},
            "mustard_0": {"description": "bottle of dijon mustard", "state": ['contains mustard'], "location": "fridge_0", "type": []},
            "whipped_cream_0": {"description": "can of whipped cream", "state": ['contains whipped_cream'], "location": "fridge_0", "type": []},
            "jam_0": {"description": "jar of mixed fruit jam", "state": ['contains mixed_fruit_jam'], "location": "fridge_0", "type": []},
            "jam_1": {"description": "jar of strawberry jam", "state": ['contains strawberry_jam'], "location": "fridge_0", "type": []},
            "spread_0": {"description": "jar of orange marmalade", "state": ['contains orange_marmalade'], "location": "fridge_0", "type": []},
            "spread_1": {"description": "jar of chocolate hazelnut spread", "state": ['contains chocolate_hazelnut_spread'], "location": "fridge_0", "type": []},
            "spread_2": {"description": "jar of peanut butter spread", "state": ['contains peanut_butter_spread'], "location": "fridge_0", "type": []},
            "spread_3": {"description": "jar of artisanal mixed nut butter spread", "state": ['contains nut_butter_spread'], "location": "fridge_0", "type": []},
            "hot_sauce_0": {"description": "bottle of habanero hot sauce", "state": ['contains habanero_hot_sauce'], "location": "fridge_0", "type": []},
            "hot_sauce_1": {"description": "bottle of tabasco hot sauce", "state": ['contains tabasco_hot_sauce'], "location": "fridge_0", "type": []},
            "bacon_0": {"description": "pack of breakfast bacon", "state": ['contains bacon_strip'], "location": "fridge_0", "type": []},
            "sausage_0": {"description": "pack of breakfast sausage", "state": ['contains sausage_link'], "location": "fridge_0", "type": []},
            "parsley": {"description": "box of fresh parsley", "state": ['contains fresh_parsley'], "location": "fridge_0", "type": []},
            "chives": {"description": "box of fresh chives", "state": ['contains fresh_chives'], "location": "fridge_0", "type": []},
            "mint": {"description": "box of fresh mint", "state": ['contains fresh_mint'], "location": "fridge_0", "type": []},
            "basil": {"description": "box of fresh basil", "state": ['contains fresh_basil'], "location": "fridge_0", "type": []},
            "lemongrass": {"description": "box of fresh lemongrass", "state": ['contains fresh_lemongrass'], "location": "fridge_0", "type": []},
            "oregano": {"description": "box of fresh oregano", "state": ['contains fresh_oregano'], "location": "fridge_0", "type": []},
            "dill": {"description": "box of fresh dill", "state": ['contains fresh_dill'], "location": "fridge_0", "type": []},
            "rosemary": {"description": "box of fresh rosemary", "state": ['contains fresh_rosemary'], "location": "fridge_0", "type": []},
            "cilantro": {"description": "a bunch of fresh cilantro", "state": ['contains fresh_cilantro'], "location": "fridge_0", "type": []},
            "agave_syrup_0": {"description": "small bottle of agave syrup", "state": ['contains agave_syrup'], "location": "fridge_0", "type": []},
            "maple_syrup_0": {"description": "small bottle of maple syrup", "state": ['contains maple_syrup'], "location": "fridge_0", "type": []},
            "maple_syrup_1": {"description": "small bottle of artificially sweetened low-calorie maple syrup", "state": ['contains low_calorie_maple_syrup'], "location": "fridge_0", "type": []},
        "drawer_0": {"description": "top kitchen drawer to left of the stove", "state": [], "location":"kitchen", "type": []},
            "spoon_0": {"description": "simple metal dinner spoon", "state": [], "location":"drawer_0", "type": []},
            "spoon_1": {"description": "simple metal dinner spoon", "state": [], "location":"drawer_0", "type": []},
            "spoon_2": {"description": "simple metal dinner spoon", "state": [], "location":"drawer_0", "type": []},
            "spoon_4": {"description": "simple metal dinner spoon", "state": [], "location":"drawer_0", "type": []},
            "fork_0": {"description": "simple metal dinner fork", "state": [], "location":"drawer_0", "type": []},
            "fork_1": {"description": "simple metal dinner fork", "state": [], "location":"drawer_0", "type": []},
            "fork_2": {"description": "simple metal dinner fork", "state": [], "location":"drawer_0", "type": []},
            "fork_4": {"description": "simple metal dinner fork", "state": [], "location":"drawer_0", "type": []},
            "knife_0": {"description": "simple metal butter knife", "state": [], "location":"drawer_0", "type": []},
            "knife_1": {"description": "simple metal butter knife", "state": [], "location":"drawer_0", "type": []},
            "knife_2": {"description": "simple metal butter knife", "state": [], "location":"drawer_0", "type": []},
            "knife_4": {"description": "simple metal butter knife", "state": [], "location":"drawer_0", "type": []},
            "knife_5": {"description": "simple steak knife", "state": [], "location":"drawer_0", "type": []},
            "knife_6": {"description": "simple steak knife", "state": [], "location":"drawer_0", "type": []},
            "knife_7": {"description": "simple steak knife", "state": [], "location":"drawer_0", "type": []},
            "knife_8": {"description": "simple steak knife", "state": [], "location":"drawer_0", "type": []},
        "drawer_1": {"description": "top kitchen drawer to right of the stove", "state": [], "location":"kitchen", "type": []},
            "ladle_0": {"description": "ladle to serve food", "state": [], "location":"drawer_1", "type": []},
            "serving_scoop_0": {"description": "scoop to serve food", "state": [], "location":"drawer_1", "type": []},
            "spatula_0": {"description": "heat-resistant small silicone spatula", "state": [], "location":"drawer_1", "type": []},
            "spatula_1": {"description": "heat-resistant large silicone spatula", "state": [], "location":"drawer_1", "type": []},
            "spatula_2": {"description": "plastic spatula", "state": [], "location":"drawer_1", "type": []},
            "spatula_3": {"description": "wooden spatula", "state": [], "location":"drawer_1", "type": []},
        "spice_cabinet_0": {"description": "small cabinet for spice bottles and related items", "state": [], "location":"kitchen", "type": []},
            "pepper_0": {"description": "small bottle of pepper", "state": ['contains pepper'], "location":"spice_cabinet_0", "type": []},
            "mixed_herbs_0": {"description": "small bottle of mixed herbs", "state": ['contains mixed_herbs'], "location":"spice_cabinet_0", "type": []},
            "paprika_0": {"description": "small bottle of paprika", "state": ['contains paprika'], "location":"spice_cabinet_0", "type": []},
            "garlic_powder_0": {"description": "small bottle of garlic powder", "state": ['contains garlic_powder'], "location":"spice_cabinet_0", "type": []},
            "onion_powder_0": {"description": "small bottle of onion powder", "state": ['contains onion_powder'], "location":"spice_cabinet_0", "type": []},
            "cayenne_0": {"description": "small bottle of cayenne", "state": ['contains cayenne'], "location":"spice_cabinet_0", "type": []},
            "cinnamon_0": {"description": "small bottle of cinnamon", "state": ['contains cinnamon'], "location":"spice_cabinet_0", "type": []},
            "nutmeg_0": {"description": "small bottle of nutmeg", "state": ['contains nutmeg'], "location":"spice_cabinet_0", "type": []},
            "cardamom_0": {"description": "small bottle of cardamom", "state": ['contains cardamom'], "location":"spice_cabinet_0", "type": []},
            "clove_0": {"description": "small bottle of clove", "state": ['contains clove'], "location":"spice_cabinet_0", "type": []},
    "dining_room": {"description": "dining room", "state": [], "location":"home", "type": []},
        "table_0": {"description": "wooden dining table with four chairs", "state": [], "location":"dining_room", "type": []},
            "napkins_0": {"description": "set of four cloth napkins", "state": [], "location":"table_0", "type": []},
            "salt_shaker_0": {"description": "table salt shaker", "state": ['contains salt'], "location":"table_0", "type": []},
            "pepper_shaker_0": {"description": "pepper shaker", "state": ['contains pepper'], "location":"table_0", "type": []},
        "chair_0": {"description": "wooden dining chair", "state": [], "location":"dining_room", "type": []},
        "chair_1": {"description": "wooden dining chair", "state": [], "location":"dining_room", "type": []},
        "chair_2": {"description": "wooden dining chair", "state": [], "location":"dining_room", "type": []},
        "chair_3": {"description": "wooden dining chair", "state": [], "location":"dining_room", "type": []},
    "bedroom": {"description": "bedroom", "state": [], "location":"home", "type": []},
        "bed_0": {"description": "queen-sized bed with a wooden frame", "state": [], "location":"bedroom", "type": []},
        "dresser_0": {"description": "wooden dresser with six drawers", "state": [], "location":"bedroom", "type": []},
        "nightstand_0": {"description": "wooden nightstand with two drawers", "state": [], "location":"bedroom", "type": []},
        "lamp_0": {"description": "bedside lamp with a white shade", "state": [], "location":"bedroom", "type": []},
        "desk_0": {"description": "work desk", "state": [], "location":"bedroom", "type": []},
    "living_room": {"description": "living room", "state": [], "location":"home", "type": []},
        "sofa_0": {"description": "three-seater sofa with a brown leather upholstery", "state": [], "location":"living_room", "type": []},
        "chair_0": {"description": "recliner with a black leather upholstery", "state": [], "location":"living_room", "type": []},
        "coffee_table_0": {"description": "wooden coffee table with a glass top", "state": [], "location":"living_room", "type": []},
        "tv_0": {"description": "flat-screen TV with a 55-inch screen", "state": [], "location":"living_room", "type": []},
    "bathroom": {"description": "bathroom", "state": [], "location":"home", "type": []},
        "toilet_0": {"description": "white porcelain toilet", "state": [], "location":"bathroom", "type": []},
        "sink_0": {"description": "white porcelain sink with a chrome faucet", "state": ['contains water'], "location":"bathroom", "type": []},
        "mirror_0": {"description": "large mirror with a wooden frame", "state": [], "location":"bathroom", "type": []},
        "shower_0": {"description": "glass-enclosed shower with a chrome showerhead", "state": [], "location":"bathroom", "type": []},
    "patio": {"description": "outdoor patio area", "state": [], "location":"home", "type": []},
        "table_1": {"description": "small patio table", "state": [], "location":"patio", "type": []},
}
entity_list = list(example_scene.keys())
entity_list.sort()

def scene_from_key(key):
    idxs = [int(x) for x in key.split('_')]
    scene = {k:v for k,v in example_scene.items() if entity_list.index(k) in idxs}
    verify_scene(scene)
    for mandatoryobj in mandatory_items:
        if mandatoryobj not in scene:
            scene[mandatoryobj] = example_scene[mandatoryobj]
    return scene

def key_from_scene(scene):
    return '_'.join(sorted([str(entity_list.index(k)) for k in scene.keys()]))

def verify_scene(scene):
    # Check that all objects are in a location
    for entity_name, entity_info in scene.items():
        assert "location" in entity_info.keys(), f"entity {entity_name} does not have a location"
        assert entity_info["location"] in scene.keys() or entity_info["location"] == 'home', f"entity {entity_name} is in a non-existent location {entity_info['location']}"
    # Track locations and see that the obejct is not repeated and that the chain ends at 'home'
    for entity_name, entity_info in scene.items():
        next_loc = entity_info["location"]
        max_depth_to_check = 20
        while (next_loc != 'home'):
            assert next_loc != entity_name, f"Found circular location chain for {entity_name}"
            next_loc = scene[next_loc]["location"]
            max_depth_to_check -= 1
            if max_depth_to_check == 0:
                raise ValueError(f"reached Max Depth! Circular location chain for {entity_name}")

class SceneGenerator():
    def __init__(self, prob=None, split=None, avoid_files = []):
        assert split is not None or prob is not None, f"Need one of probability and split"
        assert split is None or prob is None, f"Cannot do both probability and split"
        self.avoid_object_combinations = []
        self.probability = prob
        self.split_file = f"env/env_keys_{split}.txt" if split is not None else None
        self.sample = split != 'train'
        self.split_idx = 0
        if split is not None: 
            assert os.path.exists(self.split_file)
        for filepath in avoid_files:
            self.avoid_object_combinations.append([l.strip() for l in open(filepath).readlines()])

    def __call__(self):
        if self.split_file is not None:
            return self.sample_from_split()
        elif self.probability is not None:
            return self.sample_from_prob()
        else:
            raise RuntimeError("This should really never happen!!")
        
    def sample_from_split(self):
        key_options = [key.strip() for key in open(self.split_file).readlines()]
        while True:
            if self.sample: 
                key = random.choice(key_options)
            else:
                key = key_options[self.split_idx]
                self.split_idx += 1
            try:
                scene = scene_from_key(key)
                break
            except Exception as e:
                print(e)
        scene = scene_from_key(key)
        verify_scene(scene)
        return scene

    def sample_from_prob(self):
        scene = {}
        for item in example_scene:
            if item in mandatory_items:
                scene[item] = example_scene[item]
            else:
                if random.random() < self.probability:
                    scene[item] = example_scene[item]
        for _ in range(10):
            add_new = 0
            original_items = list(scene.keys())
            for item in original_items:
                if scene[item]['location'] not in scene and scene[item]['location'] != "home":
                    add_new += 1
                    scene[scene[item]['location']] = example_scene[scene[item]['location']]
            if add_new == 0: break
        verify_scene(scene)
        return scene

