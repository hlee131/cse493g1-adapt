_task_name_cereal_coffee = 'Make cereal and coffee for breakfast'
_task_name_tea_eggs = 'Make tea and eggs for breakfast'
_task_name_toast_coffee = 'Make toast and coffee for breakfast'
_task_name_toast_eggs = 'Make toast and eggs for breakfast'
_task_name_yoghurt = 'Make yoghurt parfait for breakfast'
_task_name_cereal = 'Prepare cereal for breakfast'
_task_name_eggs = 'Prepare eggs for breakfast'
_task_name_omlette = 'Prepare omelette for breakfast'
_task_name_pancake = 'Make pancakes for breakfast'
_task_name_french = 'Make a french toast for breakfast'
_task_name_coffee_oatmeal = 'Make coffee and oatmeal for breakfast'
_task_name_tea_toast = 'Prepare tea and toast for breakfast'
_task_name_scrambled = 'Prepare scrambled eggs for breakfast'

savory = '---'.join([_task_name_eggs, _task_name_omlette, _task_name_tea_eggs, _task_name_toast_eggs, _task_name_scrambled])
sweet = '---'.join([_task_name_cereal, _task_name_yoghurt, _task_name_pancake, _task_name_french, _task_name_coffee_oatmeal, _task_name_cereal_coffee])
cereal = '---'.join([_task_name_cereal_coffee, _task_name_cereal])
beverages = '---'.join([_task_name_tea_toast, _task_name_toast_coffee, _task_name_coffee_oatmeal, _task_name_cereal_coffee, _task_name_tea_eggs])
tea = '---'.join([_task_name_tea_eggs, _task_name_tea_toast])
coffee = '---'.join([_task_name_cereal_coffee, _task_name_toast_coffee, _task_name_coffee_oatmeal])
eggs = '---'.join([_task_name_eggs, _task_name_omlette, _task_name_scrambled, _task_name_tea_eggs, _task_name_toast_eggs])
toast = '---'.join([_task_name_toast_coffee, _task_name_toast_eggs, _task_name_tea_toast])
oatmeal = '---'.join([_task_name_coffee_oatmeal])
yoghurt = '---'.join([_task_name_yoghurt])
cooked = savory + '---'.join([_task_name_pancake, _task_name_french])
french_or_waffle_or_pancake = '---'.join([_task_name_french, _task_name_pancake])
french_toast = '---'.join([_task_name_french])
pancakes = '---'.join([_task_name_pancake])
food = '---'.join([_task_name_cereal_coffee, _task_name_tea_eggs, _task_name_toast_coffee, _task_name_toast_eggs, _task_name_yoghurt, _task_name_cereal, _task_name_eggs, _task_name_omlette, _task_name_pancake, _task_name_french, _task_name_coffee_oatmeal, _task_name_tea_toast, _task_name_scrambled])

manual_list = {
cooked:[
    "olive_oil",
    "spices",
    "stainless_steel, cast_iron_cookware, non_stick_pan",
    "silicone_spatula"],

eggs:[
    "omelet, omelet, poach",
    "vegan_egg_substitutes",
    "spices, herbs, parsley_chives, salt, pepper",
    "butter, no_butter, cheese, vegan_cheese",
    "condiment, chili_or_hot_sauce",
    "whole_milk, milk",
    "cheese + vegetables mixed",
    "vegetables, no veggies",
    "eggs_with_bacon + sausage hash browns, toast, fruit, nut_butter, cream, avocado"],

toast:[
    "whole_grain_toast, gluten_free_bread_for_toast",
    "no spreads, spread, butter, avocado, hummus, dairy_free_cheese, no nut butter",
    "toaster"],

beverages:[
    "black",
    "served_hot",
    "iced",
    "with_milk_and_sugar",
    "full_fat_dairy_milk",
    "honey",
    "decaf",
    "caffeinated_beverages",
    "no_plant_based_milks",
    "milk_next_to_beverage",
    "cardamom"],

coffee:[
    "espresso_machine",
    "pour over"],

tea:[
    "lemon",
    "herbal"],

cereal:[
    "gluten_free_cereals",
    "first_mix_milk_then_cereal, cereal_in_first",
    "full_fat_dairy_milk",
    "vegan milk",
    "no nut milk",
    "no_plant_based_milks",
    "plant_based_milk_with_cereal",
    "lactose_free_milk, cereal_served_with_lactose_free_milk_or_eaten_dry"],

oatmeal:[
    "water_or_water_and_plant_based_milk_if_oatmeal",
    "oatmeal_with_milk",
    "nondairy milk, lactose_free_milk"],

sweet:[
    "traditional_full_calorie_sugars",
    "honey_or_maple_syrup, with_syrup",
    "no_toppings, fresh_fruits, seed, cinnamon, cardamom, with_sweetners, with_nut, whipped_cream, fruit, honey, granola, salt",
    "milk",
    "savory toppings",
    "savory side bacon/sausage, fruits",
    "fruits -> nuts",
    "nuts -> granola"],

yoghurt:[
    "plant_based_yoghurt, no_plant_based_yogurt, flavored_yoghurt, plain_yogurt, no nut yoghurt",
    "granola",
    "fruit"],

food:[
    "beverages_before_food, beverages_prepared_first",
    "serve_at_dining_location, eat_breakfast_in_dining_area_or_kitchen, dining_room_or_patio_dining_locations, breakfast_at_desk_or_patio, breakfast_served_in_the_backyard_or_on_the_patio, serve_at_formal_dining_room_or_outdoor_patio, eat_at_standing_desk_or_kitchen_island, have_breakfast_on_kitchen_island",
    "fancy_dishes, simple_serving_dishes_and_utensils",
    "surprise_breakfast_dessert",
    "berries_as_side, nuts_as_side, fruit_or_nuts, food_at_desk"],

}

tasks = [_task_name_eggs,
_task_name_omlette,
_task_name_cereal,
_task_name_toast_coffee,
_task_name_yoghurt,
_task_name_tea_eggs,
_task_name_cereal_coffee,
_task_name_tea_toast]

reward_by_task = {task:0 for task in tasks}

for task in reward_by_task:
    for item in manual_list:
        if task in item:
            reward_by_task[task] += 1
            # reward_by_task[task] += len(manual_list[item])
    
print(reward_by_task)