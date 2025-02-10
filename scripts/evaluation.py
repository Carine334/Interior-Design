from my_app import create_app  # Import the function, not app
from models_base import User, Image, Like
from routes import content_based_filtering, hybrid_recommendations
from recommendation_engine import build_user_profile, get_liked_image_data, generate_2d_layout
from recommendation_engine import analyze_room_image
from extensions import db
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
from flask_login import current_user
from flask import request
import os
from werkzeug.utils import secure_filename
from recommendation_engine import generate_recommendations

# Create the Flask app instance
app = create_app()

# Precision at K
def precision_at_k(recommended, relevant, k=10):
    recommended_at_k = recommended[:k]
    return len(set(recommended_at_k) & set(relevant)) / k if k > 0 else 0

# Recall at K
def recall_at_k(recommended, relevant, k=10):
    recommended_at_k = recommended[:k]
    return len(set(recommended_at_k) & set(relevant)) / len(relevant) if relevant else 0

# F1-Score
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Mean Average Precision (MAP)
def mean_average_precision(users, k=10):
    avg_precisions = []
    for user in users:
        user_ratings = Like.query.filter_by(user_id=user.id).all()
        relevant = {like.image_id for like in user_ratings}
        if not relevant:
            continue
        recommended = content_based_filtering(user, Image.query.all())[:k]
        precision_values = [
            precision_at_k(recommended, relevant, i+1) 
            for i in range(len(recommended)) if recommended[i] in relevant
        ]
        avg_precisions.append(np.mean(precision_values) if precision_values else 0)
    
    return np.mean(avg_precisions) if avg_precisions else 0

# Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(users, k=10):
    reciprocal_ranks = []
    for user in users:
        user_ratings = Like.query.filter_by(user_id=user.id).all()
        relevant = {like.image_id for like in user_ratings}
        if not relevant:
            continue
        recommended = content_based_filtering(user, Image.query.all())[:k]
        for rank, img_id in enumerate(recommended, start=1):
            if img_id in relevant:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

# Normalized Discounted Cumulative Gain (NDCG)
def ndcg_at_k(recommended, relevant, k=10):
    def dcg(scores):
        return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))
    
    recommended_at_k = recommended[:k]
    gains = [1 if img_id in relevant else 0 for img_id in recommended_at_k]
    ideal_gains = sorted(gains, reverse=True)
    return dcg(gains) / dcg(ideal_gains) if dcg(ideal_gains) > 0 else 0

# Layout Evaluation Metrics
def evaluate_layout_consistency(room_dimensions, detected_furniture, empty_spaces, layout_image_path):
    furniture_area = sum([furniture["dimensions"][0] * furniture["dimensions"][1] for furniture in detected_furniture])
    room_area = room_dimensions["width"] * room_dimensions["length"]
    coverage_ratio = furniture_area / room_area if room_area > 0 else 0
    empty_space_area = sum([w * h for x, y, w, h in empty_spaces])
    unused_empty_space = empty_space_area - furniture_area
    placement_consistency = furniture_area / (furniture_area + unused_empty_space) if (furniture_area + unused_empty_space) > 0 else 0
    
    evaluation_metrics = {
        "furniture_area": furniture_area,
        "room_area": room_area,
        "coverage_ratio": coverage_ratio,
        "unused_empty_space": unused_empty_space,
        "placement_consistency": placement_consistency,
        "layout_image_path": layout_image_path
    }
    return evaluation_metrics

def evaluate_room_type_relevance(detected_furniture, room_type, layout_recommendations):
    expected_furniture = {
        "living_room": ['sofa', 'couch', 'tv', 'coffee table', 'armchair'],
        "bedroom": ['bed', 'nightstand', 'dresser', 'wardrobe', 'mirror'],
        "kitchen": ['dining table', 'chair', 'refrigerator', 'oven', 'sink'],
        "bathroom": ['toilet', 'sink', 'shower', 'bathtub']
    }
    relevant_furniture = 0
    for item in detected_furniture:
        if item["type"] in expected_furniture.get(room_type, []):
            relevant_furniture += 1

    relevance_score = relevant_furniture / len(expected_furniture.get(room_type, [])) if expected_furniture.get(room_type, []) else 0
    return relevance_score

def evaluate_color_palette(user_profile, dominant_colors):
    preferred_colors = user_profile.get("preferred_styles", [])
    common_colors = set(preferred_colors).intersection(set(dominant_colors))
    compatibility_score = len(common_colors) / len(preferred_colors) if preferred_colors else 0
    return compatibility_score

def visualize_evaluation_metrics(evaluation_metrics):
    """
    Visualize the evaluation metrics on a graph.
    Args:
        evaluation_metrics (dict): The dictionary containing evaluation metrics.
    """
    # Ensure all values are numeric (float or int)
    metrics = list(evaluation_metrics.keys())
    values = list(evaluation_metrics.values())

    # Convert non-numeric values to NaN or 0 if necessary
    values = [v if isinstance(v, (int, float)) else 0 for v in values]
    
    # Create a bar plot for evaluation metrics
    plt.figure(figsize=(10, 6))
    plt.barh(metrics, values, color='skyblue')
    plt.xlabel("Metric Value")
    plt.title("Evaluation Metrics for Room Layout")
    plt.tight_layout()
    plt.show()

# Evaluate the recommendation system
def evaluate_recommendation_system():
    with app.app_context():
        users = User.query.all()
        images = Image.query.all()

        total_users = len(users)
        precision_scores, recall_scores, f1_scores, ndcg_scores = [], [], [], []
        
        for user in users:
            user_ratings = Like.query.filter_by(user_id=user.id).all()
            relevant = {like.image_id for like in user_ratings}

            if not relevant:
                recommended_images = content_based_filtering(user, images)
            else:
                recommended_images = hybrid_recommendations(user, images)

            recommended_ids = [img.id for img in recommended_images[:10]]  # Top 10 recommendations
            
            precision = precision_at_k(recommended_ids, relevant)
            recall = recall_at_k(recommended_ids, relevant)
            f1 = f1_score(precision, recall)
            ndcg = ndcg_at_k(recommended_ids, relevant)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ndcg_scores.append(ndcg)

        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
        map_score = mean_average_precision(users)
        mrr_score = mean_reciprocal_rank(users)

        print(f"Precision@10: {avg_precision:.2%}")
        print(f"Recall@10: {avg_recall:.2%}")
        print(f"F1-Score: {avg_f1:.2%}")
        print(f"NDCG@10: {avg_ndcg:.2%}")
        print(f"Mean Average Precision (MAP): {map_score:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {mrr_score:.2%}")

# Example usage for layout evaluation with uploaded image
def evaluate_layout_example(image_path):
    """
    Evaluate the layout using real data from the uploaded image.
    Args:
        image_path (str): Path to the uploaded image.
    """
    try:
        # Analyze the uploaded image
        analysis_results = analyze_room_image(image_path)
        room_dimensions = analysis_results["room_dimensions"]
        detected_furniture = analysis_results["furniture"]
        empty_spaces = analysis_results["empty_spaces"]
        dominant_colors = analysis_results["dominant_colors"]

        # Generate the 2D layout image
        layout_image_path = generate_2d_layout(
            room_dimensions,
            detected_furniture,
            empty_spaces,
            output_path=os.path.join("static", "layouts", "generated_layout.png")
        )

        # Evaluate layout consistency
        evaluation_metrics = evaluate_layout_consistency(room_dimensions, detected_furniture, empty_spaces, layout_image_path)

        user = User.query.filter_by(id=1).first() 
        if user is None:
            raise ValueError("User with ID 1 not found")

        # Fetch user profile
        user_profile = build_user_profile(get_liked_image_data(user))
        
        # Evaluate color palette compatibility
        color_palette_score = evaluate_color_palette(user_profile, dominant_colors)

        # Visualize evaluation metrics
        visualize_evaluation_metrics(evaluation_metrics)

        print(f"Color Palette Score: {color_palette_score:.2%}")
        print(f"Evaluation Metrics: {evaluation_metrics}")

        return {
            "evaluation_metrics": evaluation_metrics,
            "color_palette_score": color_palette_score,
            "layout_image_path": layout_image_path
        }

    except Exception as e:
        print(f"Error in evaluate_layout_example: {e}")
        return None

if __name__ == "__main__":
    with app.app_context():
        # Evaluate the recommendation system
        evaluate_recommendation_system()
        # Evaluate layout with an example image
        evaluate_layout_example("C:/Users/carin/Documents/Master 2/system_recommendation/projet_final/static/uploads/fec260179e49f6abf52e34f93f1e3b98.jpg")