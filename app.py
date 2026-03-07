"""
Main Flask Application module for the Facial Recognition Service.
Provides REST API endpoints for registering users, performing facial recognition matches,
and retrieving stored user data and images.
"""
from flask import Flask, request, jsonify, send_from_directory
from config import Config
from models import db, User, FaceEncoding
from utils import extract_face_encoding
from flask_migrate import Migrate, upgrade
import os
import hashlib
from werkzeug.utils import secure_filename

app = Flask(__name__)
# Load configuration options containing the database URI
app.config.from_object(Config)

# Initialize the SQLAlchemy wrapper with the Flask app
db.init_app(app)
# Initialize Flask-Migrate for managing database schema changes
migrate = Migrate(app, db)

def save_face_image_to_disk(image_file):
    """
    Reads the file to calculate its SHA256 hash, saves it to FACE_STORAGE_DIR,
    and returns the generated filename.
    """
    # Read the data to compute hash and save
    image_data = image_file.read()
    sha256_hash = hashlib.sha256(image_data).hexdigest()
    
    # Must seek back to 0 so the extraction utility can read it again
    image_file.seek(0)
    
    # Extract original extension safely
    original_filename = secure_filename(image_file.filename)
    _, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".jpg" # fallback 
        
    new_filename = f"{sha256_hash}{ext}"
    storage_dir = app.config['FACE_STORAGE_DIR']
    
    # Ensure directory exists
    os.makedirs(storage_dir, exist_ok=True)
    
    filepath = os.path.join(storage_dir, new_filename)
    # Only write if it doesn't already exist (deduplication)
    if not os.path.exists(filepath):
        with open(filepath, 'wb') as f:
            f.write(image_data)
            
    return new_filename

@app.route('/api/v1/register', methods=['POST'])
def register_face():
    """
    Registers a new face encoding mapped to a unique identifier.
    
    Requires form-data with an 'image' file and an 'identifier' string.
    Executes strict biometric quality checks on the uploaded image.
    If successful, the image is hashed (SHA256), saved to disk, and its
    128-D encoding is saved to the PostgreSQL database.
    """
    if 'image' not in request.files or 'identifier' not in request.form:
        return jsonify({"error": "Missing 'image' or 'identifier' in form data."}), 400
        
    image_file = request.files['image']
    identifier = request.form['identifier']
    
    try:
        # Pass the Flask FileStorage object directly to our utility.
        # We enable registration_mode flag to perform strict biometric checks (resolution/blur).
        encoding, quality_score = extract_face_encoding(image_file, registration_mode=True)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
        
    # Valid face found and extracted. Now calculate hash and save to disk
    image_filename = save_face_image_to_disk(image_file)
        
    # Look up the user by identifier, or create them if they don't exist
    user = User.query.filter_by(identifier=identifier).first()
    if not user:
        user = User(identifier=identifier)
        db.session.add(user)
        db.session.flush() # Flush to assign an ID to the user before committing
        
    # Save the 128-D encoding attached to the user using pgvector, locking in their baseline
    face_record = FaceEncoding(user_id=user.id, encoding=encoding, quality_score=quality_score, image_filename=image_filename)
    db.session.add(face_record)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Database error while saving the face encoding."}), 500
        
    return jsonify({
        "status": "success",
        "message": f"Successfully registered face profile for identifier '{identifier}'."
    }), 201


@app.route('/api/v1/recognize', methods=['POST'])
def recognize_face():
    """
    Identifies a known user from an uploaded face image.
    
    Accepts form-data with an 'image' file. Extracts the face encoding and queries
    the PostgreSQL database using pgvector's L2 distance calculation for the closest match.
    
    Auto-healing Feature: If the recognized image is significantly sharper/higher quality
    than the original registration image stored in the database, the system will
    automatically overwrite the database record with the new, improved encoding and file.
    """
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' in form data."}), 400
        
    image_file = request.files['image']
    
    try:
        encoding, incoming_quality = extract_face_encoding(image_file, registration_mode=False)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
        
    # The standard threshold used by `face_recognition` library bounds is 0.6.
    # Distances < 0.6 represent a positive match; strict similarity might use a lower number like 0.4.
    MATCH_THRESHOLD = 0.6
    
    # Query PostgreSQL to find the absolute closest face encoding using pgvector's fast L2 distance calculation (`<->`)
    closest_match = FaceEncoding.query.order_by(FaceEncoding.encoding.l2_distance(encoding)).first()
    
    if closest_match:
        # Re-fetch the exact scalar distance of that closest record
        distance = db.session.query(FaceEncoding.encoding.l2_distance(encoding)).filter(FaceEncoding.id == closest_match.id).scalar()
        
        if distance < MATCH_THRESHOLD:
            response_data = {
                "match": True,
                "identifier": closest_match.user.identifier,
                "confidence_distance": round(distance, 4),
                "biometric_updated": False
            }
            
            # Auto-healing feature: If this query image is significantly higher quality/sharper 
            # than the database baseline, overwrite the baseline to continuously improve the biometrics.
            if incoming_quality > (closest_match.quality_score * 1.1) and incoming_quality > 100.0:
                # Discovered a better image! Save the file to disk by its SHA256 hash.
                new_image_filename = save_face_image_to_disk(image_file)
                
                closest_match.encoding = encoding
                closest_match.quality_score = incoming_quality
                closest_match.image_filename = new_image_filename
                try:
                    db.session.commit()
                    response_data["biometric_updated"] = True
                    response_data["biometric_improvement_message"] = "The biometric record was upgraded with this clearer facial capture."
                except Exception as e:
                    db.session.rollback()
                    # Do not fail recognition if biometric update fails
                    pass

            return jsonify(response_data), 200
            
    # If no match exists or the closest match is farther away than the threshold:
    return jsonify({
        "match": False,
        "message": "Face not recognized or no users registered."
    }), 200

@app.route('/api/v1/users', methods=['GET'])
def get_users():
    """
    Returns a list of all registered user identifiers.
    """
    users = User.query.all()
    identifiers = [u.identifier for u in users]
    return jsonify({"users": identifiers}), 200

@app.route('/api/v1/users/<identifier>/face', methods=['GET'])
def get_user_face(identifier):
    """
    Retrieves the original face image stored on disk for a given user identifier.
    
    Looks up the user's primary FaceEncoding record to find their corresponding
    SHA256-hashed image file, then serves the binary file data directly.
    """
    user = User.query.filter_by(identifier=identifier).first()
    if not user:
        return jsonify({"error": "User not found."}), 404
        
    # Find their primary face encoding record
    face_record = FaceEncoding.query.filter_by(user_id=user.id).first()
    if not face_record or not face_record.image_filename:
        return jsonify({"error": "No face image stored for this user."}), 404
        
    storage_dir = app.config['FACE_STORAGE_DIR']
    
    # Send the image file back directly
    return send_from_directory(storage_dir, face_record.image_filename)

if __name__ == '__main__':
    with app.app_context():
        # Automatically run all pending database migrations before the server starts accepting requests
        try:
            upgrade()
            print("Database migrations applied successfully.")
        except Exception as e:
            print(f"Warning: Could not apply migrations. Did you initialize the migrations folder? Error: {e}")
            
    # Run the app. Note: For production use Gunicorn or Waitress.
    app.run(host='0.0.0.0', port=5000, debug=True)
