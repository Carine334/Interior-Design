from flask import (
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    flash,
    session,
)
from extensions import db
from werkzeug.security import check_password_hash
from models_base import User, Image as ImageBase, Like
from virtual_visualization import generate_3d_render
from werkzeug.utils import secure_filename
import os
import random
from flask_login import current_user, login_required
from models_base import User
from extensions import login_manager
from flask_login import login_user
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import cv2
from flask import send_from_directory
from recommendation_engine import *
from urllib.parse import unquote
import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from skimage.transform import resize




BASE_IMAGE_PATH = r"data\data_set"

def init_routes(app):

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            if not username or not email or not password:
                flash('All fields are required!', 'danger')
                return redirect(url_for('register'))

            if User.query.filter_by(email=email).first():
                flash('Email already registered.', 'warning')
                return redirect(url_for('register'))

            new_user = User(username=username, email=email, password=password)
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful!', 'success')
            return redirect(url_for('login'))

        return render_template('register.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        next_url = request.args.get('next', url_for('for_you'))  # Redirect after login

        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            # Query the user from the database
            user = User.query.filter_by(username=username).first()

            if user and check_password_hash(user.password, password):  # Verify password hash
                # Use Flask-Login to handle the login
                login_user(user)
                
                # Store the user_id in the session
                session['user_id'] = user.id
                
                flash('Login successful!', 'success')
                return redirect(next_url)  # Redirect to the 'next' URL or 'for_you' route

            # Flash error if credentials are invalid
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login', next=next_url))

        # For GET request, render the login page
        return render_template('login.html', next=next_url)

    @app.route('/for-you')
    def for_you():
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))

        user = User.query.get(session['user_id'])
        if not user.preferences_initialized:
            return redirect(url_for('preferences'))

        # Get user's preferences
        preferences = user.preferences_set.split(',') if user.preferences_set else []
        print(f"User Preferences: {preferences}")  # Debugging: Print preferences

        # Fetch images based on style preferences
        filtered_images = ImageBase.query.filter(ImageBase.style.in_(preferences)).all()
        #print(f"Filtered Images: {filtered_images}")  # Debugging: Print filtered images

        # Check if the user has rated any images
        user_ratings = Like.query.filter_by(user_id=user.id).all()
        print(f"User Ratings: {user_ratings}")  # Debugging: Print user ratings

        if not user_ratings:
            # If the user has not rated any images, use content-based filtering only
            print("Using content-based filtering only.")
            recommended_images = content_based_filtering(user, filtered_images)
        else:
            # If the user has rated images, use hybrid recommendations
            print("Using hybrid recommendations.")
            recommended_images = hybrid_recommendations(user, filtered_images)

        # Shuffle images for randomness
        random.shuffle(recommended_images)

        # Pagination logic
        page = request.args.get('page', 1, type=int)
        per_page = 15
        total_pages = max(1, (len(recommended_images) + per_page - 1) // per_page)
        page_range = list(range(max(1, page - 2), min(total_pages + 1, page + 3)))

        # Ensure page_range is never empty
        if not page_range:
            page_range = [1]

        start = (page - 1) * per_page
        end = start + per_page
        images_on_page = recommended_images[start:end]

        return render_template(
            'for_you.html',
            images=images_on_page,
            current_page=page,
            total_pages=total_pages,
            page_range=page_range,
        )

    @app.route('/preferences', methods=['GET', 'POST'])
    def preferences():
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))

        user = User.query.get(session['user_id'])
        if request.method == 'POST':
            preferences_data = request.form.getlist('preferences')
            if len(preferences_data) > 5:
                flash('You can select up to 5 styles.', 'danger')
                return redirect(url_for('preferences'))

            user.preferences_set = ','.join(preferences_data)
            user.preferences_initialized = True  # Mark preferences as initialized
            db.session.commit()
            print(f"Saved Preferences: {user.preferences_set}")  # Debugging: Print saved preferences
            flash('Preferences saved.', 'success')
            return redirect(url_for('for_you'))

        return render_template('preferences.html')

    @app.route('/rate-image', methods=['POST'])
    def rate_image():
        if not current_user.is_authenticated:
            return jsonify({'success': False, 'message': 'User is not authenticated.'}), 401

        try:
            data = request.get_json()
            image_id = data.get('image_id')
            rating = data.get('rating')

            if not image_id or not rating:
                return jsonify({'success': False, 'message': 'Invalid data.'}), 400

            # Check if the user has already rated this image
            existing_rating = Like.query.filter_by(user_id=current_user.id, image_id=image_id).first()

            if existing_rating:
                existing_rating.rating = rating
            else:
                new_rating = Like(user_id=current_user.id, image_id=image_id, rating=rating)
                db.session.add(new_rating)

            db.session.commit()

            # Optionally, re-run recommendation generation based on updated rating
            #generate_recommendations(current_user)

            return jsonify({'success': True, 'message': 'Your vote has been recorded!'}), 200

        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'message': f"Error: {str(e)}"}), 500

    @app.route('/settings', methods=['GET', 'POST'])
    @login_required
    def settings():
        user = User.query.get(current_user.id)
        if request.method == 'POST':
            preferences_data = request.form.getlist('preferences')
            if len(preferences_data) > 5:
                flash('You can select up to 5 styles.', 'danger')
                return redirect(url_for('settings'))

            user.preferences_set = ','.join(preferences_data)
            db.session.commit()
            flash('Preferences updated.', 'success')
            return redirect(url_for('for_you'))

        preferences = user.preferences_set.split(',') if user.preferences_set else []
        return render_template('settings.html', preferences=preferences)

    @app.route('/check-auth')
    def check_auth():
        if 'user_id' not in session:
            app.logger.info('User is not authenticated')
            return '', 401  # Unauthorized
        return '', 200  # OK
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
   
    @app.route('/upload-image', methods=['GET', 'POST'])
    def upload_image_page():
        if request.method == 'POST':
            if 'room_image' not in request.files:
                flash('No file uploaded.', 'danger')
                return redirect(url_for('upload_image_page'))

            file = request.files['room_image']
            if file.filename == '':
                flash('No selected file.', 'danger')
                return redirect(url_for('upload_image_page'))

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                relative_file_path = os.path.join('uploads', filename).replace("\\", "/")
                file_path = os.path.join(app.static_folder, relative_file_path)
                file.save(file_path)
                print("Chemin d'accès: ", file_path)

                try:
                    # Generate personalized recommendations
                    recommendations = generate_recommendations(current_user, file_path)
                    print(type(recommendations))


                    # Extract recommendations
                    color_recommendations = recommendations["color_recommendations"]
                    layout_image_path = recommendations["layout_image_path"]
                    layout_image_relative_path = os.path.join('layouts', os.path.basename(layout_image_path)).replace("\\", "/")
                    layout_image_path = os.path.join(app.static_folder, layout_image_relative_path)
                    style_recommendations = recommendations["style_recommendations"]
                    furniture_recommendations = recommendations["furniture_recommendations"]
                    interactive_layout_json = recommendations["interactive_layout"].to_json() if recommendations["interactive_layout"] else None
                    similar_images = recommendations["similar_images"][:10]  # Limit to the first 10 similar images
                    similar_images_paths = []
                    list_bis = []
                    for image in similar_images[:10]:
                        relative_path_1 = os.path.join('data', 'data_set', image['image_url'])
                        file_path_ok = os.path.join(app.static_folder, relative_path_1)
                        list_bis.append(file_path_ok)
                        url = url_for('static', filename=relative_path_1.replace('\\', '/'))
                        similar_images_paths.append(url)
                    
                    similar_images_paths = [unquote(path.replace("%5C", "/")) for path in similar_images_paths]
                    similar_images_paths = [unquote(link).replace('%20', ' ') for link in similar_images_paths]

                    if isinstance(image, dict) and 'image_url' in image:
                        relative_path_1 = os.path.join('data', 'data_set', image['image_url'])
                    else:
                        print("Unexpected format:", image)

                    print("okkkkk", similar_images_paths)
                    

                    #evaluation_results = evaluate_similar_images(file_path, list_bis)
    
                    #Sort similar images by SSIM score
                    #sorted_similar_images = sorted(evaluation_results, key=lambda x: x['ssim'], reverse=True)

                    # Debug prints
                    print("Image Path:", file_path)
                    print("Layout Image Path:", layout_image_path)
                    for rec in style_recommendations:
                        print("Style Image URL:", rec["image_url"])

                    return render_template(
                        'recommendations.html',
                        image_path=url_for('static', filename=relative_file_path),
                        #similar_images=sorted_similar_images,
                        layout_image_path=url_for('static', filename=layout_image_relative_path),
                        color_recommendations=color_recommendations,
                        style_recommendations=style_recommendations,
                        furniture_recommendations=furniture_recommendations,
                        interactive_layout_json=interactive_layout_json,
                        similar_images_paths = similar_images_paths
                    )

                except Exception as e:
                    flash(f'Error generating recommendations: {str(e)}', 'danger')
                    return redirect(url_for('upload_image_page'))

        return render_template('upload_room.html')

    @app.route('/visualize-room', methods=['GET'])
    def visualize_room():
        if not current_user.is_authenticated:
            flash('Please log in to view visualizations.', 'danger')
            return redirect(url_for('login'))
        
        try:
            # Generate the 3D render
            render_path = generate_3d_render(current_user)
            return send_from_directory('static/renders', os.path.basename(render_path))
        except Exception as e:
            flash(f"Error generating visualization: {e}", 'danger')
            return redirect(url_for('upload_image_page'))
        
    @app.route('/submit-rating', methods=['POST'])
    def submit_rating():
        data = request.get_json()
        image_url = data.get('imageUrl')
        rating = data.get('rating')

        # Save the rating to the database or log it
        print(f"Received rating {rating} for image: {image_url}")

        return jsonify({"status": "success", "message": "Rating submitted successfully"})
        

   

# Recommendation Functions

def content_based_filtering(user, image_db):
    preferences = user.preferences_set.split(',') if user.preferences_set else []
    descriptions = [img.description for img in image_db]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    user_profile = " ".join(preferences)
    user_vector = vectorizer.transform([user_profile])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    ranked_images = sorted(zip(image_db, similarity_scores), key=lambda x: x[1], reverse=True)
    return [img for img, _ in ranked_images]

def matrix_factorization(user_id, image_db):
    users = User.query.all()
    images = ImageBase.query.all()
    if len(users) < 5 or len(images) < 5:
        return content_based_filtering(User.query.get(user_id), image_db)  # Fallback
    user_item_matrix = np.zeros((len(users), len(images)))
    for like in Like.query.all():
        user_item_matrix[like.user_id - 1][like.image_id - 1] = like.rating
    svd = TruncatedSVD(n_components=min(20, user_item_matrix.shape[1] - 1))
    user_item_matrix_svd = svd.fit_transform(user_item_matrix)
    user_vector = user_item_matrix_svd[user_id - 1]
    recommendations = []
    for image_id in range(len(images)):
        image_vector = user_item_matrix_svd[:, image_id]
        similarity = np.dot(user_vector, image_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(image_vector))
        recommendations.append((image_db[image_id], similarity))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [rec[0] for rec in recommendations]

def hybrid_recommendations(user, image_db):
    content_based_recs = content_based_filtering(user, image_db)
    collaborative_recs = []
    if len(Like.query.all()) >= 5:  # Only use collaborative filtering if sufficient ratings
        collaborative_recs = matrix_factorization(user.id, image_db)
    all_recommendations = list(set(content_based_recs + collaborative_recs))
    return sorted(all_recommendations, key=lambda img: img.vote_count, reverse=True)
""""
def evaluate_similar_images(original_image_path, similar_image_paths):
    original = cv2.imread(original_image_path)
    if original is None:
        print(f"Error: Unable to read the original image at {original_image_path}")
        return {'avg_mse': 0, 'avg_ssim': 0}
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    mse_scores = []
    ssim_scores = []
    
    for path in similar_image_paths:
        similar = cv2.imread(path)
        if similar is None:
            print(f"Error: Unable to read the similar image at {path}")
            continue
        
        similar = cv2.cvtColor(similar, cv2.COLOR_BGR2RGB)
        
        # Resize images to match dimensions
        h, w = original.shape[:2]
        similar = cv2.resize(similar, (w, h))
        
        # Calculate MSE
        mse = np.mean((original - similar) ** 2)
        mse_scores.append(mse)
        
        # Ensure minimum size for SSIM calculation
        min_size = 7
        if h < min_size or w < min_size:
            original_resized = cv2.resize(original, (max(w, min_size), max(h, min_size)))
            similar_resized = cv2.resize(similar, (max(w, min_size), max(h, min_size)))
        else:
            original_resized = original
            similar_resized = similar
        
        # Calculate SSIM with explicit window size and channel axis
        ssim_value = ssim(original_resized, similar_resized, win_size=min(7, min(original_resized.shape[:2])), channel_axis=-1)
        ssim_scores.append(ssim_value)

        print ('avg_mse: ',np.mean(mse_scores), 'avg_ssim: ',  np.mean(ssim_scores))
    
    return {
        'avg_mse': np.mean(mse_scores) if mse_scores else 0,
        'avg_ssim': np.mean(ssim_scores) if ssim_scores else 0
    }

"""

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS