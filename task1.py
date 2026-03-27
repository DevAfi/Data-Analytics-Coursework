import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re


def parse_iso_duration(duration_str):
    # BBC time format is like PT1H30M, so convert it to minutes first
    if not duration_str:
        return 0
    
    duration_str = duration_str.replace('PT', '')
    total_minutes = 0
    
    hours_match = re.search(r'(\d+)H', duration_str)
    if hours_match:
        total_minutes += int(hours_match.group(1)) * 60
    
    minutes_match = re.search(r'(\d+)M', duration_str)
    if minutes_match:
        total_minutes += int(minutes_match.group(1))
    
    return total_minutes

# Need prep + cook for total time
def format_total_time(prep_time, cook_time):
    prep_minutes = parse_iso_duration(prep_time) if prep_time else 0
    cook_minutes = parse_iso_duration(cook_time) if cook_time else 0
    total_minutes = prep_minutes + cook_minutes
    
    if total_minutes == 0:
        return ""
    
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    if hours == 0:
        return f"{minutes} minutes"
    elif minutes == 0:
        return f"{hours} hour{'s' if hours > 1 else ''}"
    else:
        return f"{hours} hour{'s' if hours > 1 else ''} {minutes} minutes"


# Scrape one BBC recipe page
def collect_page_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # fail fast if request is bad
        
        # Some pages had weird characters, so decode safely
        html_text = response.content.decode('utf-8', errors='replace')
        soup = BeautifulSoup(html_text, 'html.parser')
        
        # BBC recipe fields are usually in JSON-LD script tags
        scripts = soup.find_all('script', type='application/ld+json')
        recipe_data = None
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                # Sometimes recipe is direct, sometimes inside @graph
                if '@graph' in data:
                    for item in data['@graph']:
                        if item.get('@type') == 'Recipe':
                            recipe_data = item
                            break
                elif data.get('@type') == 'Recipe':
                    recipe_data = data
                
                if recipe_data:
                    break
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        if not recipe_data:
            raise ValueError("Recipe data not found in page")
        
        # Pull all fields needed for the required output columns
        title = recipe_data.get('name') or recipe_data.get('headline', '')
        
        # Convert prep/cook into one readable total-time string
        prep_time = recipe_data.get('prepTime', '')
        cook_time = recipe_data.get('cookTime', '')
        total_time = format_total_time(prep_time, cook_time)
        
        # Image can be dict/list/string depending on the page
        image_obj = recipe_data.get('image', {})
        if isinstance(image_obj, dict):
            image = image_obj.get('url', '')
        elif isinstance(image_obj, list) and len(image_obj) > 0:
            image = image_obj[0].get('url', '') if isinstance(image_obj[0], dict) else image_obj[0]
        else:
            image = str(image_obj) if image_obj else ''
        
        # Keep ingredients in one comma-separated string
        ingredients_list = recipe_data.get('recipeIngredient', [])
        ingredients = ', '.join(ingredients_list) if ingredients_list else ''
        
        # Ratings usually live under aggregateRating
        aggregate_rating = recipe_data.get('aggregateRating', {})
        rating_val = aggregate_rating.get('ratingValue', '')
        rating_count = aggregate_rating.get('ratingCount', '')
        
        # Category/cuisine tags
        category = recipe_data.get('recipeCategory', '')
        cuisine = recipe_data.get('recipeCuisine', '')
        
        # Diet field can be string or list, so normalize it
        suitable_for_diet = recipe_data.get('suitableForDiet', [])
        if not isinstance(suitable_for_diet, list):
            suitable_for_diet = [suitable_for_diet] if suitable_for_diet else []
        
        # Quick boolean flags for vegan/vegetarian
        vegan = 'VeganDiet' in str(suitable_for_diet) or 'vegan' in str(suitable_for_diet).lower()
        vegetarian = 'VegetarianDiet' in str(suitable_for_diet) or 'vegetarian' in str(suitable_for_diet).lower()
        
        # Make diet names readable instead of raw schema links
        diet_list = []
        for diet in suitable_for_diet:
            if isinstance(diet, str):
                diet_name = diet.split('/')[-1].replace('Diet', '').title()
                diet_list.append(diet_name)
        diet = ', '.join(diet_list) if diet_list else ''
        
        # Build dataframe with the exact requested column set
        data = {
            'title': [title],
            'total_time': [total_time],
            'image': [image],
            'ingredients': [ingredients],
            'rating_val': [rating_val],
            'rating_count': [rating_count],
            'category': [category],
            'cuisine': [cuisine],
            'diet': [diet],
            'vegan': [vegan],
            'vegetarian': [vegetarian],
            'url': [url]
        }
        
        df = pd.DataFrame(data)
        
        return df
        
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        # Return empty df with same schema so later code does not break.
        columns = ['title', 'total_time', 'image', 'ingredients', 'rating_val', 'rating_count', 
                  'category', 'cuisine', 'diet', 'vegan', 'vegetarian', 'url']
        return pd.DataFrame(columns=columns)
    
    except Exception as e:
        print(f"Error processing recipe data: {e}")
        # Same fallback for any unexpected parsing issue
        columns = ['title', 'total_time', 'image', 'ingredients', 'rating_val', 'rating_count', 
                  'category', 'cuisine', 'diet', 'vegan', 'vegetarian', 'url']
        return pd.DataFrame(columns=columns)


if __name__ == "__main__":
    test_urls = [
        'https://www.bbc.co.uk/food/recipes/easiest_ever_banana_cake_42108',
        'https://www.bbc.co.uk/food/recipes/vegetablecurry_80763',
        'https://www.bbc.co.uk/food/recipes/quick_butter_saag_with_85874',
        'https://www.bbc.co.uk/food/recipes/cod_and_chorizo_stew_91004',
        'https://www.bbc.co.uk/food/recipes/slow_cooker_aubergine_51283',
        'https://www.bbc.co.uk/food/recipes/satay_sweet_potato_curry_59527',
        'https://www.bbc.co.uk/food/recipes/steak_diane_with_saut_67797',
    ]
    
    print("Testing collect_page_data function...")
    print("=" * 60)
    
    all_results = []

    for url in test_urls:
        print(f"\nProcessing: {url}")
        df = collect_page_data(url)
        if not df.empty:
            all_results.append(df)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nDataFrame contents:")
        print(df.to_string())
        print("\n" + "=" * 60)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        csv_filename = 'recipe_data.csv'
        combined_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"\nCSV file '{csv_filename}' has been generated successfully.")
    else:
        print("\nNo recipe data was collected; CSV not generated.")
