import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, current_user, login_required
from flask_cors import CORS
from datetime import timedelta
from extensions import db
from models_base import User, Image, Like
from routes import init_routes
from sqlalchemy import any_
from flask_migrate import Migrate


def create_app():
    # Set up Flask app
    basedir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.abspath(r'C:\Users\carin\Documents\Master 2\system_recommendation\projet_final\templates')
    static_folder = os.path.abspath(r'C:\Users\carin\Documents\Master 2\system_recommendation\projet_final\static')
    app = Flask(__name__, template_folder=template_dir, static_folder=static_folder)

    # Flask configuration
    instance_path = os.path.join(basedir, 'instance')
    db_path = os.path.join(instance_path, 'interior_designer.sqlite')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}?timeout=10'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'ok'
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
    app.config['UPLOAD_FOLDER'] = r'C:\Users\carin\Documents\Master 2\system_recommendation\projet_final\static\uploads'

    # Initialize database
    db.init_app(app)

    # Initialize LoginManager
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access this page.'

    # Initialize Flask-Migrate
    migrate = Migrate(app, db)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register routes
    init_routes(app)

    # Create instance directory if not exists
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    # Create database tables
    with app.app_context():
        db.create_all()

    # CORS setup
    CORS(app, supports_credentials=True)


    # Render the home page
    @app.route('/')
    def home():
        return render_template('register.html')  # Your main landing page

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
