# Import dependencies
import json
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from extensions import db
from werkzeug.security import generate_password_hash
from flask_login import UserMixin
from sqlalchemy.orm import relationship

# Define models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), nullable=False, unique=True)  
    password = db.Column(db.String(200), nullable=False)
    interest_vector = db.Column(db.JSON)
    preferences_set = db.Column(db.JSON, nullable=True)  # Store preferences as a JSON array
    preferences_initialized = db.Column(db.Boolean, default=False)  # New field
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = generate_password_hash(password)
    
    def __repr__(self):
        return f'<User {self.username}>'
    


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    style = db.Column(db.String(50))
    room_type = db.Column(db.String(50)) 
    tokens =  db.Column(db.Text)
    furniture_type = db.Column(db.String(50))  # Add this column
    sustainability_score = db.Column(db.Float)
    adaptability_score = db.Column(db.Float)

    likes = relationship('Like', backref='image', lazy='dynamic')

    @property
    def vote_count(self):
        return self.likes.count()

class Like(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    rating = db.Column(db.Float)

class DesignRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_url = db.Column(db.String(255), nullable=False)
    room_type = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
