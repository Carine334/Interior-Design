import json
import os
from flask import Flask
from extensions import db  # Import your app and Image model
from models_base import Image

# Initialize Flask app
app = Flask(__name__)

# Flask configuration
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
db_path = os.path.join(instance_path, 'interior_designer.sqlite')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key_here'

db.init_app(app)

# JSON file path
json_file_path = r"C:\Users\carin\Documents\Master 2\system_recommendation\projet_final\static\data\concatenated_data.json"

# Predefined furniture categories based on room types
furniture_keywords =  ["cabinet", "stove", "oven", "sink", "refrigerator", "counter", "dishwasher",
     "bed", "wardrobe", "dresser", "nightstand", "mirror", "closet",
     "sofa", "chair", "coffee table", "bookshelf", "TV stand", "ottoman","bathtub", "toilet", 
    "shower", "sink", "vanity", "mirror","dining table", "dining chair", "buffet", "sideboard", "china cabinet"]

# Function to extract furniture types from description
def extract_furniture_types(description):
    if not description:
        return set()
    description_lower = description.lower()
    return set(item for item in furniture_keywords if item in description_lower)

# Function to extract id, style, and room_type from the image path
def extract_id_style_and_room_type(image_path):
    parts = image_path.split("\\")  # For Windows-style paths
    if len(parts) >= 3:
        style = parts[0]
        room_type = parts[1]
        file_name = parts[2]
        id = int(file_name.split(".")[0])  # Convert to integer
        return id, style, room_type
    else:
        return None, None, None

# Populate the database
with app.app_context():
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Update data in Image table
    for record in data:
        id, style, room_type = extract_id_style_and_room_type(record['image_path'])

        # Extract unique furniture types
        furniture_types = extract_furniture_types(record.get('text'))
        furniture_serialized = json.dumps(list(furniture_types))

        # Update existing record
        image = Image.query.get(id)
        if image:
            image.furniture_type = furniture_serialized
            db.session.commit()

print("Furniture types updated with unique entries.")
