## Create some object lists to be used in the evaluation functions. Each object in the list is a dictionary with two keys: "id" and "description"

## Create a master list of different object categories with their descriptions

list_of_all_rooms = {
    kitchen: 'kitchen',
    dining_room: 'dining room',
    living_room: 'living room',
    bathroom: 'bathroom',
    bedroom: 'bedroom',
}

list_of_all_loose_food_items = {
    tomato_0:'roma tomato',
    tomato_1:'roma tomato',
    tomato_2:'roma tomato',
    bell_pepper_0:'green bell pepper',
    bell_pepper_1:'red bell pepper',
    onion_0:'yellow onion',
    potato_0:'sweet potato',
    garlic_0:'clove of garlic',
    jalepeno_0:'spicy jalepeno peppers',
    cucumber_0:'cucumber',
    lettuce_0:'head of iceberg lettuce',
    lemon_0:'a lemon sliced in half',
    banana_0:'a ripe banana',
    apple_0:'a honeycrisp apple',
    apple_1:'a green apple',
    egg_0:'brown egg',
    egg_1:'brown egg',
    egg_2:'brown egg',
    egg_3:'brown egg',
    egg_4:'brown egg',
    egg_5:'brown egg',
    egg_6:'brown egg',
    egg_7:'brown egg',
}

list_of_all_objects_containing_food_items = {
    salt_box_0: 'box of iodized salt',
    oil_bottle_0: 'bottle of olive oil',
    oil_bottle_1: 'bottle of vegetable oil',
    bread_0: 'a sliced loaf of white sandwich bread',
    bread_1: 'a loaf of sprouted whole grain bread',
    bread_2: 'a sliced loaf of sweet banana and walnut bread',
    cereal_box_0: 'carton of fruit loops',
    cereal_box_1: 'carton of whole wheat shreds',
    cereal_box_2: 'carton of light swiss muesli',
    cereal_box_3: 'gluten-free seeds and nuts granola',
    cereal_box_4: 'a box of cocoa puffs',
    cereal_box_5: 'a box of classic corn flakes',
    oatmeal_0: 'packet of overnight oats',
    oatmeal_1: 'box of Quaker 1-minute instant oats',
    oatmeal_2: 'plain rolled oats',
    sugar_0: 'small bottle of white sugar',
    sugar_1: 'packet of brown cane sugar',
    sugar_2: 'small bottle of loose calorie-free stevia',
    honey: 'bottle of honey',
    tea_bags_0: 'box of green tea bags',
    tea_bags_1: 'box of herbal peppermint tea bags',
    tea_bags_2: 'box of chamomile tea',
    tea_bags_3: 'box of earl grey tea bags',
    tea_bags_4: 'box of english breakfast tea bags',
    coffee_0: 'bag of regular ground coffee',
    coffee_1: 'bag of decaf ground coffee',
    coffee_2: 'jar of instant coffee',
    milk_0: 'carton of whole dairy milk',
    milk_1: 'carton of oat milk',
    milk_2: 'tin of condensed milk',
    milk_3: 'bottle of almond milk',
    milk_4: 'jug of skim dairy milk',
    yoghurt_0: 'pack of plain non-fat greek yoghurt',
    yoghurt_1: 'cup of full-fat raspberry yoghurt with raspberry jam',
    yoghurt_2: 'cup of vegan cashew-based yoghurt',
    yoghurt_3: 'bottle of probiotic drinkable yoghurt',
    egg_carton_0: 'carton of 12 eggs',
    liquid_egg_0: 'a box of vegan liquid egg substitute',
    butter_0: 'block of butter',
    butter_1: 'box of vegan butter',
    cheese_0: 'slices of american cheese',
    cheese_1: 'block of part skim mozzarella cheese',
    cheese_2: 'pack of goat cheese',
    cheese_3: 'block of aged cheddar cheese',
    cheese_4: 'slices of vegan cheese',
    blueberries_box_0: 'a pint-sized box of organic blueberries',
    strawberries_box_0: 'a pint-sized box of organic raspberries',
    kiwi_box_0: 'a box of sweet golden kiwis',
    oranges_bag_0: 'a bag of navel oranges',
    oranges_bag_1: 'a bag of blood oranges',
    mixed_berries_0: 'a bag of frozen mixed berries',
    ketchup_0: 'bottle of ketchup',
    mayonnaise_0: 'jar of mayonnaise',
    mustard_0: 'bottle of dijon mustard',
    hot_sauce_0: 'bottle of habanero hot sauce',
    hot_sauce_1: 'bottle of tabasco hot sauce',
    pepper_0: 'small bottle of pepper',
    mixed_herbs_0: 'small bottle of mixed herbs',
    paprika_0: 'small bottle of paprika',
    garlic_powder_0: 'small bottle of garlic powder',
    onion_powder_0: 'small bottle of onion powder',
    cayenne_0: 'small bottle of cayenne',
    cinnamon_0: 'small bottle of cinnamon',
    nutmeg_0: 'small bottle of nutmeg',
    cardamom_0: 'small bottle of cardamom',
    clove_0: 'small bottle of clove',
    salt_shaker_0: 'table salt shaker',
    pepper_shaker_0: 'pepper shaker',
}

list_of_all_kitchenware = {
    knife_0: 'simple metal butter knife',
    knife_1: 'simple metal butter knife',
    knife_2: 'simple metal butter knife',
    knife_3: 'bread knife with a serrated edge',
    scissor_0: 'a pair of kitchen scissors',
    bowl_0: 'microwaveable glass bowl',
    bowl_1: 'microwaveable glass bowl',
    bowl_2: 'microwaveable glass bowl',
    bowl_3: 'microwaveable glass bowl',
    bowl_4: 'white ceramic salad bowl',
    bowl_5: 'white ceramic salad bowl',
    bowl_6: 'white ceramic salad bowl',
    bowl_7: 'white ceramic salad bowl',
    bowl_8: 'fancy porcelain china bowl with pink flowers',
    bowl_9: 'fancy porcelain china bowl with pink flowers',
    bowl_10: 'small steel bowl',
    bowl_11: 'small steel bowl',
    plate_0: 'flat white ceramic plate',
    plate_1: 'flat white ceramic plate',
    plate_2: 'flat white ceramic plate',
    plate_3: 'flat white ceramic plate',
    plate_4: 'deep white ceramic plate',
    plate_5: 'deep white ceramic plate',
    plate_6: 'deep white ceramic plate',
    plate_7: 'deep white ceramic plate',
    plate_8: 'fancy porcelain china plate with pink flowers',
    plate_9: 'fancy porcelain china plate with pink flowers',
    cup_0: 'fancy porcelain teacup with saucer',
    cup_1: 'fancy porcelain teacup with saucer',
    cup_2: 'simple white ceramic cup',
    cup_3: 'simple white ceramic cup',
    mug_0: 'plain white mug',
    mug_1: 'plain white mug',
    french_press_0: 'a glass french press',
    pour_over_coffee_maker_0: 'a chemex pour over coffee maker',
    glass_0: 'tall glass for drinking water',
    glass_1: 'tall glass for drinking water',
    glass_2: 'tall glass for drinking water',
    glass_3: 'tall glass for drinking water',
    glass_4: 'short fancy cocktail glass',
    glass_5: 'short fancy cocktail glass',
    glass_6: 'short fancy cocktail glass',
    glass_7: 'short fancy cocktail glass',
    wine_glass_0: 'standard wine glass',
    wine_glass_1: 'standard wine glass',
    wine_glass_2: 'standard wine glass',
    wine_glass_3: 'standard wine glass',
    wine_glass_4: 'tall champagne flute',
    wine_glass_5: 'tall champagne flute',
    wine_glass_6: 'tall champagne flute',
    wine_glass_7: 'tall champagne flute',
    pan_0: 'classic non-stick pan',
    pan_1: 'metal pan',
    skillet_0: 'cast iron skillet',
    pot_0: 'small aluminium pot',
    pot_1: 'large aluminium pot',
    cutting_board_1: 'wooden cutting board',
    spoon_0: 'simple metal dinner spoon',
    spoon_1: 'simple metal dinner spoon',
    spoon_2: 'simple metal dinner spoon',
    spoon_4: 'simple metal dinner spoon',
    fork_0: 'simple metal dinner fork',
    fork_1: 'simple metal dinner fork',
    fork_2: 'simple metal dinner fork',
    fork_4: 'simple metal dinner fork',
    knife_4: 'simple metal butter knife',
    knife_5: 'simple steak knife',
    knife_6: 'simple steak knife',
    knife_7: 'simple steak knife',
    knife_8: 'simple steak knife',
    ladle_0: 'ladle to serve food',
    serving_scoop_0: 'scoop to serve food',
    spatula_0: 'heat-resistant small silicone spatula',
    spatula_1: 'heat-resistant large silicone spatula',
    spatula_2: 'plastic spatula',
    spatula_3: 'wooden spatula',
    napkins_0: 'set of four cloth napkins',
}

list_of_all_cleaning_supplies = {
    sponge_0: 'sponge for cleaning',
    dish_soap_0: 'bottle of dish soap',
}

list_of_all_appliances = {
    oven_0: 'oven',
    microwave_0: 'microwave',
    kettle_0: 'an electric kettle',
    toaster_0: 'a classic 2-bread toaster',
    dishwasher_0: 'dishwasher',
    coffee_machine_0: 'a drip coffee machine',
    espresso_machine_0: 'an espresso machine',
    french_press_0: 'a glass french press',
    pour_over_coffee_maker_0: 'a chemex pour over coffee maker',
    fridge_0: 'fridge',
}

list_of_all_furniture_and_surfaces = {
    island_0: 'kitchen island next to countertop with two seats',
    island_chair_0: 'bar height chair next to kitchen island',
    island_chair_1: 'bar height chair next to kitchen island',
    cabinet_0: 'left kitchen cabinet above the counter',
    cabinet_1: 'right kitchen cabinet above the counter',
    cabinet_2: 'tall kitchen cabinet',
    cabinet_3: 'kitchen cabinet below the counter',
    counter_0: 'kitchen countertop next to the stove',
    counter_1: 'kitchen countertop next to the sink',
    chair_0: 'recliner with a black leather upholstery',
    chair_1: 'wooden dining chair',
    chair_2: 'wooden dining chair',
    chair_3: 'wooden dining chair',
    bed_0: 'queen-sized bed with a wooden frame',
    dresser_0: 'wooden dresser with six drawers',
    nightstand_0: 'wooden nightstand with two drawers',
    table_0: 'wooden table with four chairs',
    sofa_0: 'three-seater sofa with a brown leather upholstery',
    coffee_table_0: 'wooden coffee table with a glass top',
    sink_1: 'standard kitchen sink',
}


## Create semantically meaningful lists of objects from the above master lists that are useful in experessing various preferences

list_of_coffee_makers = [coffee_machine_0, espresso_machine_0, french_press_0, pour_over_coffee_maker_0]

list_of_spices = [pepper_0, mixed_herbs_0, paprika_0, garlic_powder_0, onion_powder_0, cayenne_0, cinnamon_0, nutmeg_0, cardamom_0, clove_0, salt_shaker_0, pepper_shaker_0]

list_of_healthy_foods = [tomato_0, tomato_1, tomato_2, bell_pepper_0, bell_pepper_1, onion_0, potato_0, garlic_0, jalepeno_0, cucumber_0, lettuce_0, lemon_0, banana_0, apple_0, apple_1, egg_0, egg_1, egg_2, egg_3, egg_4, egg_5, egg_6, egg_7, oil_bottle_0, oil_bottle_1, bread_1, cereal_box_1, cereal_box_2, cereal_box_3, cereal_box_5, oatmeal_0, oatmeal_1, oatmeal_2, sugar_2, honey, tea_bags_0, tea_bags_1, tea_bags_2, coffee_0, coffee_1, coffee_2, milk_1, milk_3, milk_4, yoghurt_0, yoghurt_2, yoghurt_3, liquid_egg_0, butter_1, cheese_1, cheese_2, cheese_3, cheese_4, blueberries_box_0, strawberries_box_0, kiwi_box_0, oranges_bag_0, oranges_bag_1, mixed_berries_0, mustard_0, hot_sauce_0, hot_sauce_1, mixed_herbs_0, paprika_0, garlic_powder_0, onion_powder_0, cayenne_0, cinnamon_0, nutmeg_0, cardamom_0, clove_0]

list_of_low_fat_foods = [yoghurt_0, milk_4, oatmeal_0, oatmeal_1, oatmeal_2, tea_bags_0, tea_bags_1, tea_bags_2, coffee_0, coffee_1, coffee_2, mixed_herbs_0, paprika_0, garlic_powder_0, onion_powder_0, cayenne_0, cinnamon_0, nutmeg_0, cardamom_0, clove_0, tomato_0, tomato_1, tomato_2, bell_pepper_0, bell_pepper_1, onion_0, potato_0, garlic_0, jalepeno_0, cucumber_0, lettuce_0, lemon_0, banana_0, apple_0, apple_1, blueberries_box_0, strawberries_box_0, kiwi_box_0, oranges_bag_0, oranges_bag_1, mixed_berries_0]

list_of_non_vegetarian_foods = [egg_0, egg_1, egg_2, egg_3, egg_4, egg_5, egg_6, egg_7]

list_of_non_vegan_foods = [egg_0, egg_1, egg_2, egg_3, egg_4, egg_5, egg_6, egg_7, milk_0, milk_1, milk_2, milk_3, milk_4, yoghurt_0, yoghurt_1, yoghurt_2, yoghurt_3, butter_0, cheese_0, cheese_1, cheese_2, cheese_3, mayonnaise_0]

list_of_foods_containing_caffeine = [coffee_0, coffee_1, coffee_2, tea_bags_3, tea_bags_4]

list_of_foods_containing_gluten = [bread_0, bread_1, bread_2, cereal_box_1]

list_of_foods_containing_lactose = [milk_0, milk_2, milk_4]

list_of_vegetables = [tomato_0, tomato_1, tomato_2, bell_pepper_0, bell_pepper_1, onion_0, potato_0, garlic_0, jalepeno_0, cucumber_0, lettuce_0, lemon_0]

list_of_fruits = [banana_0, apple_0, apple_1, blueberries_box_0, strawberries_box_0, kiwi_box_0, oranges_bag_0, oranges_bag_1, mixed_berries_0]

list_of_condiments = [ketchup_0, mayonnaise_0, mustard_0, hot_sauce_0, hot_sauce_1]

list_of_heat_resistant_kitchenwares = [bowl_0, bowl_1, bowl_2, bowl_3, bowl_4, bowl_5, bowl_6, bowl_7, plate_0, plate_1, plate_2, plate_3, plate_4, plate_5, plate_6, plate_7, cup_2, cup_3, mug_0, mug_1, ladle_0, serving_scoop_0, spatula_0, spatula_1, spatula_3, glass_0, glass_1, glass_2, glass_3, pan_0, pan_1, skillet_0, pot_0, pot_1]

list_of_durable_kitchenwares = [ladle_0, serving_scoop_0, spatula_0, spatula_1, spatula_3, bowl_0, bowl_1, bowl_2, bowl_3, bowl_4, bowl_5, bowl_6, bowl_7, plate_0, plate_1, plate_2, plate_3, plate_4, plate_5, plate_6, plate_7, cup_2, cup_3, mug_0, mug_1, glass_0, glass_1, glass_2, glass_3, skillet_0]

list_of_eco_friendly_kitchenwares = [spatula_3, bowl_0, bowl_1, bowl_2, bowl_3, glass_0, glass_1, glass_2, glass_3, glass_4, glass_5, glass_6, glass_7, wine_glass_0, wine_glass_1, wine_glass_2, wine_glass_3, wine_glass_4, wine_glass_5, wine_glass_6, wine_glass_7, skillet_0, pot_0, pot_1]

list_of_fragile_expensive_kitchenwares = [bowl_8, bowl_9, plate_8, plate_9, cup_0, cup_1, glass_4, glass_5, glass_6, glass_7, wine_glass_0, wine_glass_1, wine_glass_2, wine_glass_3, wine_glass_4, wine_glass_5, wine_glass_6, wine_glass_7]

list_of_silverware = [spoon_0, spoon_1, spoon_2, spoon_4, fork_0, fork_1, fork_2, fork_4, knife_4, knife_5, knife_6, knife_7, knife_8]
