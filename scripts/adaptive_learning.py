from models import Image

def generate_adaptive_space_recommendations(room_type):
    adaptive_furniture = Image.query.filter_by(room_type=room_type).order_by(
        Image.adaptability_score.desc()).limit(5).all()
    
    recommendations = [{
        'id': furniture.id,
        'url': furniture.url,
        'description': furniture.description,
        'furniture_type': furniture.furniture_type,
        'adaptability_score': furniture.adaptability_score
    } for furniture in adaptive_furniture]
    
    return recommendations
