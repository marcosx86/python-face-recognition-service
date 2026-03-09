"""
Main Flask Application module for the Facial Recognition Service.
Provides REST API endpoints for registering users, performing facial recognition matches,
and retrieving stored user data and images.
"""
# Standard library
import os
import hashlib
import logging
import argparse
import threading
from datetime import datetime, timezone, timedelta

# Third-party
from flask import Flask, request, jsonify, send_from_directory
from flask_migrate import Migrate, upgrade
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from waitress import serve
import requests

# Local
from config import Config
from models import db, User, FaceEncoding, Webhook, DetectionLog
from utils import extract_face_encoding

logger = logging.getLogger(__name__)

# Lock to serialize CPU-heavy face processing (dlib is NOT thread-safe).
# Concurrent calls into dlib's C++ code cause segfaults / silent crashes.
# Initialized at startup.
_face_processing_lock = threading.Lock()

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
    # Ensure stream is at the beginning before reading (it may have been consumed by extraction)
    image_file.seek(0)
    
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
    logger.debug(f"Generated new filename: {new_filename} from original: {original_filename}")
    storage_dir = app.config['FACE_STORAGE_DIR']
    
    # Ensure directory exists
    os.makedirs(storage_dir, exist_ok=True)
    
    filepath = os.path.join(storage_dir, new_filename)
    # Only write if it doesn't already exist (deduplication)
    if not os.path.exists(filepath):
        with open(filepath, 'wb') as f:
            f.write(image_data)
            
    return new_filename

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Global exception handler to capture unhandled errors that might break the server.
    Logs the full traceback to our telemetry/console.
    """
    # Pass through standard HTTP errors (like 404, 400)
    logger.debug(f"Exception caught at application level: {e}")
    if isinstance(e, HTTPException):
        return e

    # Log the full exception traceback
    logger.error(f"Unhandled Exception caught at application level: {e}", exc_info=True)
    return jsonify({"error": "An unexpected Internal Server Error occurred."}), 500

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
        logger.warning("Missing 'image' or 'identifier' in form data during registration.")
        return jsonify({"error": "Missing 'image' or 'identifier' in form data."}), 400

    if not _face_processing_lock.acquire(blocking=False):
        logger.warning("Server busy — rejecting registration request.")
        return jsonify({"error": "Server is busy processing other requests. Try again shortly."}), 503

    try:
        image_file = request.files['image']
        identifier = request.form['identifier']
        logger.info(f"Processing registration request for identifier: '{identifier}'")
    
        try:
            # Pass the Flask FileStorage object directly to our utility.
            # We enable registration_mode flag to perform strict biometric checks (resolution/blur).
            encoding, quality_score = extract_face_encoding(image_file, registration_mode=True)
        except ValueError as e:
            logger.warning(f"Registration failed: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Unknown exception during registration: {e}")
            return jsonify({"error": str(e)}), 500
        
        try:
            # Valid face found and extracted. Now calculate hash and save to disk
            image_filename = save_face_image_to_disk(image_file)
        except Exception as e:
            logger.error(f"Unknown exception during image saving: {e}")
            return jsonify({"error": str(e)}), 500
        
        # Look up the user by identifier, or create them if they don't exist
        user = User.query.filter_by(identifier=identifier).first()
        if not user:
            user = User(identifier=identifier)
            db.session.add(user)
            db.session.flush() # Flush to assign an ID to the user before committing
        
        # Save the 128-D encoding attached to the user using pgvector, locking in their baseline
        logger.debug(f"Saving face encoding and quality score ({quality_score:.1f}) to database for user ID: {user.id}")
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
    finally:
        _face_processing_lock.release()


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
        logger.warning("Missing 'image' in form data during recognition.")
        return jsonify({"error": "Missing 'image' in form data."}), 400

    if not _face_processing_lock.acquire(blocking=False):
        logger.warning("Server busy — rejecting recognition request.")
        return jsonify({"error": "Server is busy processing other requests. Try again shortly."}), 503

    try:
        image_file = request.files['image']
        logger.info("Processing recognition request.")
    
        try:
            encoding, incoming_quality = extract_face_encoding(image_file, registration_mode=False)
        except ValueError as e:
            logger.warning(f"Recognition failed: {e}")
            return jsonify({"error": str(e)}), 400
        
        # The standard threshold used by `face_recognition` library bounds is 0.6.
        # Distances < 0.6 represent a positive match; strict similarity might use a lower number like 0.4.
        MATCH_THRESHOLD = 0.6
    
        # Query PostgreSQL to find the absolute closest face encoding using pgvector's fast L2 distance calculation (`<->`)
        closest_match = FaceEncoding.query.order_by(FaceEncoding.encoding.l2_distance(encoding)).first()
    
        if closest_match:
            # Re-fetch the exact scalar distance of that closest record
            distance = db.session.query(FaceEncoding.encoding.l2_distance(encoding)).filter(FaceEncoding.id == closest_match.id).scalar()
            logger.debug(f"Closest match distance for ID {closest_match.user.id}: {distance}")
        
            if distance < MATCH_THRESHOLD:
                logger.info(f"Face recognized as '{closest_match.user.identifier}' with confidence distance: {distance:.4f}")
                response_data = {
                    "match": True,
                    "identifier": closest_match.user.identifier,
                    "confidence_distance": round(distance, 4),
                    "biometric_updated": False
                }
            
                # Webhook dispatch and Logging logic
                now = datetime.now(timezone.utc)
            
                # Log the detection
                detection = DetectionLog(user_id=closest_match.user.id, timestamp=now)
                db.session.add(detection)
            
                # Rate limiting webhooks (1 minute)
                should_dispatch_webhooks = False
                last_seen = closest_match.user.last_seen_time
                # Ensure last_seen is timezone-aware for comparison with `now` (which is UTC-aware)
                try:
                    if last_seen is not None and last_seen.tzinfo is None:
                        last_seen = last_seen.replace(tzinfo=timezone.utc)
                    if last_seen is None or (now - last_seen) > timedelta(minutes=1):
                        should_dispatch_webhooks = True
                except Exception as e:
                    logger.error(f"Failed to process should_dispatch_webhooks: {e}")
                
                closest_match.user.last_seen_time = now
                # We must commit these DB changes now even if webhooks fail
                try:
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    logger.error(f"Failed to log detection: {e}")

                if should_dispatch_webhooks:
                    # Find webhooks for this user
                    webhooks = Webhook.query.filter_by(user_id=closest_match.user.id).all()
                    for wh in webhooks:
                        payload = {
                            "event": wh.event_kind,
                            "identifier": closest_match.user.identifier,
                            "confidence_distance": round(distance, 4)
                        }
                        try:
                            logger.debug(f"Sending webhook to {wh.target_url} for '{closest_match.user.identifier}'")
                            # Send synchronously (timeout=3s to avoid hanging the API too long)
                            requests.post(wh.target_url, json=payload, timeout=3.0)
                        except Exception as e:
                            logger.error(f"Failed to dispatch webhook to {wh.target_url}: {e}")
            
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
        logger.info("Face not recognized or distance above threshold.")
        return jsonify({
            "match": False,
            "message": "Face not recognized or no users registered."
        }), 200
    finally:
        _face_processing_lock.release()

@app.route('/api/v1/webhooks', methods=['POST'])
def register_webhook():
    """
    Registers a new webhook URL for a specific user and event kind.
    Expected form or JSON data: 'identifier', 'event_kind', 'target_url'
    """
    data = request.json or request.form
    if not data or 'identifier' not in data or 'event_kind' not in data or 'target_url' not in data:
        return jsonify({"error": "Missing 'identifier', 'event_kind', or 'target_url'."}), 400
        
    identifier = data['identifier']
    event_kind_name = data['event_kind']
    target_url = data['target_url']
    
    # 1. Verify User
    user = User.query.filter_by(identifier=identifier).first()
    if not user:
        return jsonify({"error": f"User with identifier '{identifier}' not found."}), 404
        
    # 2. Create Webhook directly
    logger.info(f"Registering new webhook for user '{identifier}' ({event_kind_name}) at {target_url}")
    webhook = Webhook(user_id=user.id, event_kind=event_kind_name, target_url=target_url)
    db.session.add(webhook)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to save webhook to database."}), 500
        
    return jsonify({
        "status": "success",
        "message": "Webhook registered successfully.",
        "webhook_id": webhook.id
    }), 201

@app.route('/api/v1/webhooks', methods=['GET'])
def list_webhooks():
    """
    Lists all registered webhooks. Optionally filter by ?identifier=<id>.
    """
    identifier = request.args.get('identifier')
    query = db.session.query(Webhook).join(User)
    
    if identifier:
        query = query.filter(User.identifier == identifier)
        
    webhooks = query.all()
    
    results = []
    for wh in webhooks:
        results.append({
            "id": wh.id,
            "identifier": wh.user.identifier,
            "event_kind": wh.event_kind,
            "target_url": wh.target_url
        })
        
    return jsonify({"webhooks": results}), 200

@app.route('/api/v1/webhooks/<int:webhook_id>', methods=['DELETE'])
def delete_webhook(webhook_id):
    """
    De-registers (deletes) a webhook given its ID.
    """
    webhook = Webhook.query.get(webhook_id)
    if not webhook:
        return jsonify({"error": "Webhook not found."}), 404
        
    db.session.delete(webhook)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to delete webhook."}), 500
        
    return jsonify({"status": "success", "message": "Webhook deleted successfully."}), 200

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
    parser = argparse.ArgumentParser(description="Facial Recognition Service API")
    parser.add_argument('--loglevel', default='INFO', help='Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--debug-webserver', action='store_true', help='Enable Flask debug mode and Werkzeug debugger')
    args = parser.parse_args()

    with app.app_context():
        # Automatically run all pending database migrations before the server starts accepting requests
        try:
            upgrade()
        except Exception as e:
            # Since we do not have our logger configured yet, we print the error to the console.
            print(f"Could not apply migrations. Did you initialize the migrations folder? Error: {e}")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    
    # Apply our logging config after Alembic's upgrade().
    # Alembic's env.py calls fileConfig() which overwrites the root logger level to WARN.
    logging.basicConfig(level=numeric_level,
                        force=True,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("Face processing lock initialized (dlib is not thread-safe, serializing access).")
            
    # Use Waitress as the production WSGI server; fall back to Werkzeug only for debugging
    try:
        if args.debug_webserver:
            logger.warning("Starting Werkzeug development server with debugger enabled.")
            app.run(host='0.0.0.0', port=5000, debug=True)
        else:
            logger.info("Starting Waitress production server on http://0.0.0.0:5000")
            serve(app, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Failed to run server: {e}")
