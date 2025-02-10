def calculate_sustainability_score(recommendations):
    eco_friendly_materials = ["bamboo", "recycled plastic", "reclaimed wood"]
    energy_efficient_items = ["LED lights", "smart thermostat", "solar panels"]
    
    score = 0
    for item in recommendations:
        if item["furniture_type"] in eco_friendly_materials:
            score += 1
        if item["furniture_type"] in energy_efficient_items:
            score += 1
    
    return (score / len(recommendations)) * 100 if recommendations else 0
