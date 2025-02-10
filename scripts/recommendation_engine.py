import numpy as np
from scipy.spatial.distance import cosine
from models_base import User, Image, Like
import matplotlib.pyplot as plt
import cv2
import torch
import sys
from pathlib import Path
from sklearn.cluster import KMeans
import bpy
import cv2
import numpy as np
from collections import Counter
import spacy
from matplotlib.patches import Rectangle, Circle
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
from flask import flash
import matplotlib.patches as patches


# Load spaCy's English model for NLP
nlp = spacy.load("en_core_web_sm")

def estimate_room_dimensions(image):
    """
    Estimate room dimensions from an image by detecting the floor and walls.
    Args:
        image (numpy.ndarray): Input room image.
    Returns:
        dict: Estimated dimensions of the room in arbitrary units (width, length, height).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    width, length, height = 0, 0, 0
    
    if lines is not None:
        vertical_lines = []
        horizontal_lines = []
        for rho, theta in lines[:, 0]:
            if np.abs(theta) < np.pi / 4 or np.abs(theta - np.pi) < np.pi / 4:
                vertical_lines.append(rho)
            elif np.abs(theta - np.pi / 2) < np.pi / 4:
                horizontal_lines.append(rho)
        
        if vertical_lines and horizontal_lines:
            width = max(vertical_lines) - min(vertical_lines)
            length = max(horizontal_lines) - min(horizontal_lines)
    
    height = 2.5  # Assume a fixed height
    return {"width": abs(width), "length": abs(length), "height": height}


def analyze_room_image(image_path):
    """
    Analyze the room image to detect furniture, empty spaces, room type, and dominant colors.
    Args:
        image_path (str): Path to the room image.
    Returns:
        dict: Analysis results including furniture, empty spaces, room type, and dominant colors.
    """
    image = cv2.imread(image_path)
    room_dimensions = estimate_room_dimensions(image)
    furniture = detect_furniture(image)  # Ensure this returns a list
    empty_spaces = detect_empty_spaces(image)  # Ensure this returns a list
    room_type = infer_room_type(furniture)
    dominant_colors = detect_dominant_colors(image)  # Ensure this returns a list of colors

    return {
        "furniture": furniture,  # Must be a list
        "empty_spaces": empty_spaces,  # Must be a list
        "room_type": room_type,
        "dominant_colors": dominant_colors,  # Must be a list
        "room_dimensions": room_dimensions
    }


def detect_empty_spaces(image, min_area=1000):
    """
    Detect empty spaces in the room image.
    Args:
        image (numpy.ndarray): Input room image.
        min_area (int): Minimum area of an empty space to consider.
        
    Returns:
        list: List of empty spaces as bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours based on area
    empty_spaces = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area:  # Only consider spaces larger than min_area
            empty_spaces.append((x, y, w, h))
    
    return empty_spaces

def load_yolov5_model():
    """
    Load the YOLOv5 model for furniture detection.
    Returns:
        torch model: Pretrained YOLOv5 model.
    """
    yolov5_path = Path("C:/Users/carin/Documents/Master 2/system_recommendation/projet_final/scripts/yolov5")
    sys.path.append(str(yolov5_path))
    return torch.hub.load(str(yolov5_path), 'yolov5s', source='local', pretrained=True)

def detect_furniture(image):
    """
    Detect furniture in the room image using YOLOv5.
    Args:
        image (numpy.ndarray): Input room image.
    Returns:
        list: List of detected furniture items with type, position, dimensions, and confidence.
    """
    try:
        model = load_yolov5_model()
        results = model(image)
        detected_furniture = []
        furniture_classes = ['chair', 'couch', 'bed', 'dining table', 'tv', 'sofa', 'armchair', 'coffee table', 'bathroom vanity', 'toilet', 
                             'shower', 'bathtub', 'medicine cabinet', 'kitchen cabinet', 'kitchen island', 'bar stool', 'refrigerator', 'stove',
                            'oven', 'dishwasher', 'microwave', 'dresser', 'nightstand', 'wardrobe', 'bookshelf', 'desk', 'ottoman', 'end table', 
                            'tv stand', 'entertainment center', 'sideboard', 'buffet', 'china cabinet', 'dining chair', 'bench', 'recliner', 'loveseat',
                            'console table', 'floor lamp', 'table lamp', 'mirror', 'rug', 'curtains', 'blinds']    
        
        for detection in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, confidence, class_id = detection
            class_name = model.names[int(class_id)]
            if class_name in furniture_classes:
                detected_furniture.append({
                    "type": class_name,
                    "position": ((x1 + x2) // 2, (y1 + y2) // 2),  # Center of the bounding box
                    "dimensions": (x2 - x1, y2 - y1),  # Width and height of the bounding box
                    "confidence": confidence
                })
        
        print("The detected furnitures are: ", detected_furniture)
        return detected_furniture
    
    except Exception as e:
        print(f"Error detecting furniture: {e}")
        return []

def infer_room_type(detected_furniture):
    """
    Infer the room type based on detected furniture.
    Args:
        detected_furniture (list): List of detected furniture items.
    Returns:
        str: Inferred room type (e.g., "living_room", "bedroom").
    """
    room_type_mapping = {
        "living_room": ['sofa', 'couch', 'tv', 'coffee table', 'armchair', 'bookshelf', 'rug', 'lamp', 'tv stand'],
        "bedroom": ['bed', 'nightstand', 'dresser', 'wardrobe', 'mirror', 'lamp', 'dresser', 'bedside table'],
        "kitchen": ['dining table', 'chair', 'refrigerator', 'oven', 'sink', 'microwave', 'dishwasher', 'counter', 'cabinet'],
        "bathroom": ['toilet', 'sink', 'shower', 'bathtub', 'mirror', 'towel rack', 'toilet paper holder', 'bath mat'],
        "dining_room": ['dining table', 'chairs', 'sideboard', 'buffet', 'china cabinet', 'light fixture']
    }

    room_type_scores = {room: 0 for room in room_type_mapping.keys()}
    for item in detected_furniture:
        for room, furniture_list in room_type_mapping.items():
            if item["type"] in furniture_list:
                room_type_scores[room] += 1

        
    return max(room_type_scores, key=room_type_scores.get)

def detect_dominant_colors(image, num_colors=5):
    """
    Detect dominant colors in the room image.
    Args:
        image (numpy.ndarray): Input room image.
        num_colors (int): Number of dominant colors to detect.
    Returns:
        list: List of dominant colors in HEX format.
    """
    resized_image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = resized_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

def get_liked_image_data(user):
    """
    Fetch tokens and styles from images liked by the user.
    Args:
        user (User): The current user object.
    Returns:
        list: List of dictionaries containing tokens and styles of liked images.
    """
    liked_images = Like.query.filter_by(user_id=user.id).all()
    liked_image_ids = [like.image_id for like in liked_images]
    liked_images_data = Image.query.filter(Image.id.in_(liked_image_ids)).all()
    
    # Extract tokens and styles from liked images
    liked_data = []
    for img in liked_images_data:
        liked_data.append({
            "tokens": img.tokens,  # Assuming tokens are stored as a list in the database
            "style": img.style     # Assuming style is stored as a string in the database
        })
    return liked_data

def extract_features(descriptions):
    """
    Extract features (styles, colors, themes, furniture) from descriptions.
    Args:
        descriptions (list): List of image descriptions.
    Returns:
        dict: Extracted features with counts.
    """
    keywords, themes, styles, colors, furniture = [], [], [], [], []
    
    for desc in descriptions:
        doc = nlp(desc)
        for token in doc:
            if token.pos_ == "ADJ":
                styles.append(token.text)
            elif token.pos_ == "NOUN":
                if token.text in ["room", "bedroom", "kitchen", "living room"]:
                    themes.append(token.text)
                else:
                    furniture.append(token.text)
            elif token.text in ["gray", "white", "warm"]:  # Example color list
                colors.append(token.text)
        
        for chunk in doc.noun_chunks:
            if "table" in chunk.text or "sofa" in chunk.text or "bed" in chunk.text:
                furniture.append(chunk.text)
    
    return {
        "keywords": Counter(keywords).most_common(5),
        "themes": Counter(themes).most_common(3),
        "styles": Counter(styles).most_common(3),
        "colors": Counter(colors).most_common(3),
        "furniture": Counter(furniture).most_common(5)
    }

def build_user_profile(liked_data):
    """
    Build a user profile based on tokens and styles from liked images.
    Args:
        liked_data (list): List of dictionaries containing tokens and styles of liked images.
    Returns:
        dict: User profile with preferred styles, colors, themes, and furniture.
    """
    styles = []
    tokens = []
    
    for data in liked_data:
        styles.append(data["style"])
        tokens.extend(data["tokens"])  # Assuming tokens are already preprocessed
    
    # Count the most common styles and tokens
    style_counter = Counter(styles)
    token_counter = Counter(tokens)
    
    return {
        "preferred_styles": [style for style, _ in style_counter.most_common(3)],
        "preferred_keywords": [token for token, _ in token_counter.most_common(5)]
    }

# Define category-based color mapping
FURNITURE_COLORS = {
    "bedroom": "#66b3ff",
    "living_room": "#ff9999",
    "kitchen": "#99ff99",
    "bathroom": "#ffcc99",
    "storage": "#c2c2f0",
    "decor": "#ffb3e6",
    "default": "#b3b3b3"
}

# Define furniture categories
FURNITURE_CATEGORIES = {
    "bedroom": ["bed", "nightstand", "wardrobe", "dresser"],
    "living_room": ["couch", "sofa", "armchair", "coffee table", "recliner", "loveseat", "tv stand", "entertainment center"],
    "kitchen": ["dining table", "bar stool", "kitchen cabinet", "kitchen island", "refrigerator", "stove", "oven", "dishwasher", "microwave"],
    "bathroom": ["bathroom vanity", "toilet", "shower", "bathtub", "medicine cabinet"],
    "storage": ["bookshelf", "sideboard", "buffet", "china cabinet"],
    "decor": ["console table", "mirror", "rug", "curtains", "blinds", "table lamp", "floor lamp"]
}

def get_furniture_color(furniture_type):
    """Returns the color based on the furniture type category."""
    for category, items in FURNITURE_CATEGORIES.items():
        if furniture_type in items:
            return FURNITURE_COLORS[category]
    return FURNITURE_COLORS["default"]

def generate_2d_layout(room_dimensions, detected_furniture, empty_spaces=None, output_path="static/layouts/generated_layout.png"):
    """
    Generate a realistic 2D layout with furniture visualization.
    
    Args:
        room_dimensions (dict): Width, length, and height of the room.
        detected_furniture (list): List of detected furniture items.
        empty_spaces (list): List of empty spaces as bounding boxes.
        output_path (str): Path to save the generated layout image.
    
    Returns:
        str: Path to the saved layout image.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, room_dimensions["width"])
        ax.set_ylim(0, room_dimensions["length"])
        ax.set_aspect('equal')

        # Set background color for the room
        ax.set_facecolor("#f0f0f0")

        # Draw the room boundaries (walls)
        wall_thickness = 0.2  # 20cm walls
        ax.add_patch(patches.Rectangle((0, 0), room_dimensions["width"], room_dimensions["length"], 
                                       edgecolor='black', facecolor='#d9d9d9', linewidth=3))

        # Draw grid lines for better scaling
        grid_spacing = 1  # 1 meter spacing
        for x in np.arange(0, room_dimensions["width"], grid_spacing):
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.5)
        for y in np.arange(0, room_dimensions["length"], grid_spacing):
            ax.axhline(y, color="gray", linestyle="--", linewidth=0.5)

        # Draw detected furniture
        if detected_furniture:
            for furniture in detected_furniture:
                x, y = furniture["position"]
                width, height = furniture["dimensions"]
                furniture_type = furniture["type"].lower()
                color = get_furniture_color(furniture_type)

                # Different shapes for different furniture types
                if furniture_type in ["chair", "stool", "table lamp", "floor lamp"]:
                    # Use circles for small furniture
                    ax.add_patch(patches.Circle((x + width / 2, y + height / 2), width / 2, edgecolor="black", facecolor=color, alpha=0.8))
                elif furniture_type in ["sofa", "couch", "bed", "armchair", "recliner", "loveseat"]:
                    # Use rounded rectangles for seating and beds
                    ax.add_patch(patches.FancyBboxPatch(
                        (x, y), width, height, boxstyle="round,pad=0.1",
                        edgecolor="black", facecolor=color, alpha=0.8
                    ))
                else:
                    # Default to rectangles for other furniture
                    ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor="black", facecolor=color, alpha=0.8))

                # Add label with furniture type
                ax.text(x + width / 2, y + height / 2, 
                        f"{furniture_type.capitalize()}\n({width:.1f}cm x {height:.1f}cm)", 
                        ha='center', va='center', fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Draw empty spaces (if provided)
        if empty_spaces:
            for space in empty_spaces:
                x, y, w, h = space
                ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='green', 
                                               facecolor='lightgreen', alpha=0.4, linestyle="dashed"))
                ax.text(x + w / 2, y + h / 2, f"Empty Space\n({w:.1f}cm x {h:.1f}cm)", 
                        ha='center', va='center', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Save the plot as an image
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()  # Close the plot to free up memory

        print(f"Saving layout to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error generating 2D layout: {e}")
        return None


def generate_interactive_layout(room_dimensions, detected_furniture, empty_spaces=None):
    """
    Generate an interactive 2D layout using Plotly.
    Args:
        room_dimensions (dict): Width, length, and height of the room.
        detected_furniture (list): List of detected furniture items.
        empty_spaces (list): List of empty spaces as bounding boxes.
    Returns:
        go.Figure: Interactive Plotly figure.
    """
    fig = go.Figure()

    # Draw room boundaries
    fig.add_trace(go.Scatter(
        x=[0, room_dimensions["width"], room_dimensions["width"], 0, 0],
        y=[0, 0, room_dimensions["length"], room_dimensions["length"], 0],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Room Boundaries"
    ))

    # Draw furniture
    for furniture in detected_furniture:
        x, y = furniture["position"]
        width, height = furniture["dimensions"]
        fig.add_trace(go.Scatter(
            x=[x, x + width, x + width, x, x],
            y=[y, y, y + height, y + height, y],
            mode="lines+text",
            fill="toself",
            line=dict(color="blue", width=2),
            fillcolor="lightblue",
            opacity=0.7,
            name=furniture["type"],
            text=[f"{furniture['type'].capitalize()}<br>({width:.2f}m x {height:.2f}m)"],
            textposition="middle center"
        ))

    # Draw empty spaces
    if empty_spaces:
        for space in empty_spaces:
            x, y, w, h = space
            fig.add_trace(go.Scatter(
                x=[x, x + w, x + w, x, x],
                y=[y, y, y + h, y + h, y],
                mode="lines+text",
                fill="toself",
                line=dict(color="green", width=2),
                fillcolor="lightgreen",
                opacity=0.4,
                name="Empty Space",
                text=[f"Empty Space<br>({w:.2f}m x {h:.2f}m)"],
                textposition="middle center"
            ))

    # Update layout
    fig.update_layout(
        title="Interactive 2D Room Layout",
        xaxis_title="Width (meters)",
        yaxis_title="Length (meters)",
        showlegend=True,
        template="plotly_white"
    )

    return fig

def generate_personalized_layout(user, image_path):
    """
    Generate a personalized 2D layout based on the user's profile and preferences.
    Args:
        user (User): The current user object.
        image_path (str): Path to the room image.
    Returns:
        str: Path to the saved layout image.
    """
    # Analyze the room image
    analysis_results = analyze_room_image(image_path)
    
    # Fetch liked image data and build user profile
    liked_data = get_liked_image_data(user)
    user_profile = build_user_profile(liked_data)
    
    # Generate layout recommendations based on user preferences
    layout_recommendations = get_layout_recommendations(analysis_results["room_type"], analysis_results["furniture"], analysis_results["empty_spaces"])
    
    # Generate the 2D layout image
    layout_image_path = generate_2d_layout(
        analysis_results["room_dimensions"],
        analysis_results["furniture"],
        analysis_results["empty_spaces"],
        output_path=f"C:/Users/carin/Documents/Master 2/system_recommendation/projet_final/static/layouts/generated_layout_{user.id}.png"
    )
    
    return layout_image_path

def get_layout_recommendations(room_type, furniture, empty_spaces):
    """
    Generate layout recommendations based on room type, furniture, and empty spaces.
    Args:
        room_type (str): Type of the room (e.g., "living_room", "bedroom").
        furniture (list): List of detected furniture items.
        empty_spaces (list): List of empty spaces as bounding boxes.
    Returns:
        list: List of layout recommendations.
    """
    layout_templates = {
        "living_room": [
            "Place sofa along the main wall.",
            "Add a coffee table centrally.",
            "Position the TV opposite the sofa."
        ],
        "bedroom": [
            "Place the bed in the center of the room.",
            "Add nightstands on either side of the bed.",
            "Position the wardrobe against the wall."
        ],
        "kitchen": [
            "Place the dining table in the center.",
            "Position the refrigerator in a corner.",
            "Arrange the sink and stove in a functional layout."
        ],
        "bathroom": [
            "Place the sink opposite the door.",
            "Position the toilet next to the sink.",
            "Arrange the shower in a corner."
        ]
    }
    return layout_templates.get(room_type, [])

def get_color_palette_recommendations(user_profile, dominant_colors):
    """
    Generate color palette recommendations based on user preferences and dominant colors.
    Args:
        user_profile (dict): User profile with preferred styles.
        dominant_colors (list): List of dominant colors in the room image.
    Returns:
        list: List of recommended colors in HEX format.
    """
    color_palettes = {
        "Art Deco": ["#000000", "#FFD700", "#003366", "#008000", "#800020", "#C0C0C0", "#F5F5DC", "#36454F"],
        "Bohemian": ["#E2725B", "#FFDB58", "#800080", "#008080", "#FF7F50", "#808000", "#F5F5DC", "#FFFFFF"],
        "Coastal": ["#87CEEB", "#FFFFFF", "#F4A460", "#20B2AA", "#FF6F61", "#000080", "#D3D3D3", "#8B4513"],
        "Contemporary": ["#FFFFFF", "#000000", "#808080", "#FF0000", "#FFFF00", "#0000FF", "#F5F5DC", "#36454F"],
        "Eclectic": ["#4169E1", "#FFD700", "#8B0000", "#40E0D0", "#FFA500", "#32CD32", "#FFFFFF", "#808080"],
        "Farmhouse": ["#FFFFFF", "#F5F5DC", "#D3D3D3", "#556B2F", "#000080", "#CC5500", "#8B4513", "#C0C0C0"],
        "Mediterranean": ["#E2725B", "#00BFFF", "#FFFFFF", "#808000", "#FFD700", "#8B0000", "#F5F5DC", "#A9A9A9"],
        "Mid-Century Modern": ["#FFDB58", "#808000", "#008080", "#FFA500", "#8B4513", "#FF69B4", "#FFFFFF", "#808080"],
        "Rustic": ["#8B4513", "#F5F5DC", "#808080", "#006400", "#CC5500", "#8B0000", "#C0C0C0", "#36454F"],
        "Traditional": ["#000080", "#8B0000", "#006400", "#FFD700", "#F5F5DC", "#808080", "#FFFFFF", "#36454F"],
        "Transitional": ["#808080", "#F5F5DC", "#FFFFFF", "#000080", "#008000", "#FFD700", "#A9A9A9", "#36454F"],
    }
    
    preferred_styles = user_profile.get("preferred_styles", [])
    
    recommended_colors = []
    for style in preferred_styles:
        recommended_colors.extend(color_palettes.get(style, []))
    
    complementary_colors = get_complementary_colors(dominant_colors)
    return list(set(recommended_colors) | set(complementary_colors))


def get_complementary_colors(colors):
    """
    Generate complementary colors for a given list of colors.
    Args:
        colors (list): List of colors in HEX format.
    Returns:
        list: List of complementary colors in HEX format.
    """
    complementary_colors = []
    for color in colors:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        complementary_colors.append(f"#{255 - r:02x}{255 - g:02x}{255 - b:02x}")
    return complementary_colors

def visualize_color_palette(colors):
    """
    Visualize the recommended color palette.
    Args:
        colors (list): List of colors in HEX format.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()

def generate_recommendations(user, image_path):
    """
    Generate personalized recommendations for the user.
    Args:
        user (User): The current user object.
        image_path (str): Path to the room image.
    Returns:
        dict: Recommendations including color, layout, style, furniture, and interactive layout.
    """
    try:
        # Analyze the room image
        analysis_results = analyze_room_image(image_path)

        # Fetch liked image data and build user profile
        liked_data = get_liked_image_data(user)
        user_profile = build_user_profile(liked_data)

        # Generate color recommendations
        color_recommendations = get_color_palette_recommendations(user_profile, analysis_results.get("dominant_colors", []))

        # Generate layout recommendations
        layout_image_path = generate_personalized_layout(user, image_path)

        # Ensure the layout image path is valid
        if layout_image_path is None or not os.path.exists(layout_image_path):
            flash('Failed to generate layout recommendations.', 'warning')
            layout_image_path = None  # Set to None if the layout image is not available

        # Generate style recommendations
        style_recommendations = get_style_recommendations(user, analysis_results.get("room_type", "unknown"))

        # Generate furniture recommendations
        furniture_recommendations = get_furniture_recommendations(
            user,
            analysis_results.get("empty_spaces", []),
            analysis_results.get("furniture", []),
            analysis_results.get("room_type", "unknown")
        )

        # Fetch images from the database safely
        image_db = Image.query.all() or []  # Ensure it's a list

        # Generate similar images
        similar_images = recommend_similar_images(
            user,
            analysis_results.get("furniture", []),  # Provide default empty list
            analysis_results.get("dominant_colors", []),  # Provide default empty list
            image_db
        )

        # Generate interactive layout
        interactive_layout = generate_interactive_layout(
            analysis_results.get("room_dimensions", {}),
            analysis_results.get("furniture", []),
            analysis_results.get("empty_spaces", [])
        )

        # Return all recommendations
        return {
            "similar_images": similar_images,
            "analysis_results": analysis_results,
            "color_recommendations": color_recommendations,
            "layout_image_path": layout_image_path,
            "style_recommendations": style_recommendations,
            "furniture_recommendations": furniture_recommendations,
            "interactive_layout": interactive_layout,  # Add interactive layout to the response
            "user_profile": user_profile
        }

    except Exception as e:
        print(f"Error in generate_recommendations: {e}")
        return {
            "analysis_results": {},
            "color_recommendations": [],
            "layout_image_path": None,
            "style_recommendations": [],
            "furniture_recommendations": [],
            "interactive_layout": None,  # Return None for interactive layout in case of error
            "user_profile": {},
            "similar_images": []  # Return empty list for similar images
        }


def get_style_recommendations(user, room_type):
    preferences = user.preferences_set.split(',') if user.preferences_set else []
    liked_images = Like.query.filter_by(user_id=user.id).all()
    liked_image_ids = [like.image_id for like in liked_images]
    liked_images_data = Image.query.filter(Image.id.in_(liked_image_ids)).all()
    return [{"style": img.style, "description": img.description, "image_url": img.url} for img in liked_images_data if img.room_type == room_type and img.style in preferences]



def get_furniture_recommendations(user, empty_spaces, detected_furniture, room_type):
    """
    Generate furniture recommendations based on user preferences, empty spaces, and detected furniture.
    Args:
        user (User): The current user object.
        empty_spaces (list): List of empty spaces as bounding boxes.
        detected_furniture (list): List of detected furniture items.
        room_type (str): Type of the room (e.g., "living_room", "bedroom").
    Returns:
        list: List of furniture recommendations.
    """
    if not isinstance(detected_furniture, list):
        detected_furniture = []
    if not isinstance(empty_spaces, list):
        empty_spaces = []
    
    preferences = user.preferences_set.split(',') if user.preferences_set else []
    recommendations = []
    liked_images = Like.query.filter_by(user_id=user.id).all()
    liked_image_ids = [like.image_id for like in liked_images]
    liked_images_data = Image.query.filter(Image.id.in_(liked_image_ids)).all()
    
    for img in liked_images_data:
        if img.room_type == room_type:
            furniture_type = detected_furniture[0]["type"] if len(detected_furniture) > 0 else "general"
            position = empty_spaces[0] if len(empty_spaces) > 0 else None
            
            recommendations.append({
                "style": img.style,
                "description": img.description,
                "furniture_type": furniture_type,
                "position": position
            })
    
    return recommendations

def calculate_furniture_similarity(detected_furniture, image_furniture):
    """
    Calculate the similarity between detected furniture and furniture in an image.
    
    Args:
        detected_furniture (list): List of detected furniture items.
        image_furniture (list): List of furniture items in an image.
    
    Returns:
        float: Similarity score (0 to 1).
    """
    if detected_furniture is None:
        detected_furniture = []
    if image_furniture is None:
        image_furniture = []

    detected_furniture_types = set([f["type"] for f in detected_furniture])
    image_furniture_types = set(image_furniture)
    
    intersection = detected_furniture_types.intersection(image_furniture_types)
    union = detected_furniture_types.union(image_furniture_types)
    
    return len(intersection) / len(union) if union else 0

def calculate_color_similarity(dominant_colors, image_colors):
    """
    Calculate the similarity between dominant colors and colors in an image.
    
    Args:
        dominant_colors (list): List of dominant colors in HEX format.
        image_colors (list): List of colors in an image in HEX format.
    
    Returns:
        float: Similarity score (0 to 1).
    """
    if dominant_colors is None:
        dominant_colors = []
    if image_colors is None:
        image_colors = []

    dominant_colors_set = set(dominant_colors)
    image_colors_set = set(image_colors)
    
    intersection = dominant_colors_set.intersection(image_colors_set)
    union = dominant_colors_set.union(image_colors_set)
    
    return len(intersection) / len(union) if union else 0

def extract_colors_from_description(description):
    """
    Extract colors from an image description using NLP techniques.
    
    Args:
        description (str): The textual description of the image.
    
    Returns:
        list: Extracted colors from the description.
    """
    if description is None:
        return []

    known_colors = {"red", "blue", "green", "yellow", "white", "black", "gray", 
                    "brown", "pink", "purple", "orange", "beige", "gold", "silver"}

    words = description.lower().split()  # Basic tokenization
    extracted_colors = [word for word in words if word in known_colors]

    return extracted_colors

def recommend_similar_images(user, detected_furniture, dominant_colors, image_db):
    """
    Recommend images with similar furniture and colors to the user.
    
    Args:
        user (User): The current user object.
        detected_furniture (list): List of detected furniture items.
        dominant_colors (list): List of dominant colors in the room.
        image_db (list): List of Image objects.
    
    Returns:
        list: List of recommended images with similarity scores.
    """
    # Ensure inputs are valid
    if detected_furniture is None:
        detected_furniture = []
    if dominant_colors is None:
        dominant_colors = []
    if image_db is None:
        image_db = []

    recommended_images = []
    
    for image in image_db:
        # Ensure image has a description and furniture_type
        if not hasattr(image, "description") or not hasattr(image, "furniture_type"):
            continue

        # Extract colors from the description
        extracted_colors = extract_colors_from_description(image.description)
        
        # Calculate furniture similarity
        furniture_similarity = calculate_furniture_similarity(detected_furniture, image.furniture_type)
        
        # Calculate color similarity
        color_similarity = calculate_color_similarity(dominant_colors, extracted_colors)
        
        # Combine similarity scores (adjust weights if necessary)
        combined_similarity = 0.5 * furniture_similarity + 0.5 * color_similarity
        
        recommended_images.append({
            "image_id": image.id,
            "image_url": image.url,
            "furniture_similarity": furniture_similarity,
            "color_similarity": color_similarity,
            "combined_similarity": combined_similarity
        })
    
    # Sort images by combined similarity score
    recommended_images.sort(key=lambda x: x["combined_similarity"], reverse=True)
    
    return recommended_images