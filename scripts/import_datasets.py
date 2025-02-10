from models_base import Image
import json
from extensions import db

def import_mmis_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for item in data:
        new_image = Image(
            url=item['image_url'],
            description=item['description'],
            room_type=item['room_type']
        )
        db.session.add(new_image)
    db.session.commit()

def import_adssfid_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    for item in data:
        image = Image.query.filter_by(url=item['image_url']).first()
        if image:
            image.aesthetic_score = item['aesthetic_score']
            image.decoration_style = item['decoration_style']
    db.session.commit()

# Run import functions
import_mmis_dataset('path/to/mmis_dataset.json')
import_adssfid_dataset('path/to/adssfid_dataset.json')
