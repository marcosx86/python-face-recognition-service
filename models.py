"""
SQLAlchemy database models for the Facial Recognition Service.
Defines the User schema and their associated FaceEncodings utilizing
PostgreSQL's pgvector extension for efficient similarity matching.
"""
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """
    Represents a registered individual in the system.
    Users are identified by a unique string (e.g., employee ID, email)
    and can have one or more biometric face encodings attached to their profile.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    # The identifier could be an employee ID, username, or email
    identifier = db.Column(db.String(255), unique=True, nullable=False)
    
    register_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_seen_time = db.Column(db.DateTime, nullable=True)
    
    # One-to-many relationship: A user can have multiple face encodings (e.g., different profiles/lighting)
    encodings = db.relationship('FaceEncoding', backref='user', lazy=True, cascade="all, delete-orphan")
    webhooks = db.relationship('Webhook', backref='user', lazy=True, cascade="all, delete-orphan")
    detection_logs = db.relationship('DetectionLog', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.identifier}>"

class FaceEncoding(db.Model):
    """
    Stores 128-dimensional facial vectors and quality metadata.
    Utilizes pgvector for high-performance L2 distance similarity search.
    """
    __tablename__ = 'face_encodings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Standard face_recognition (dlib) encodings are 128-dimensional vectors
    encoding = db.Column(Vector(128), nullable=False)
    
    # Track the blurriness/sharpness score so we can auto-improve the record later
    quality_score = db.Column(db.Float, nullable=False, default=0.0)
    
    # Store the SHA256 filename of the original face picture
    image_filename = db.Column(db.String(255), nullable=True)
    
class DetectionLog(db.Model):
    """
    Audit trail of every time a person was detected/recognized successfully.
    """
    __tablename__ = 'detection_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<DetectionLog User {self.user_id} at {self.timestamp}>"

class Webhook(db.Model):
    """
    Stores webhook URLs registered to a particular user.
    """
    __tablename__ = 'webhooks'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    event_kind = db.Column(db.String(50), nullable=False)
    target_url = db.Column(db.Text, nullable=False)
    
    def __repr__(self):
        return f"<Webhook {self.id} for User {self.user_id} Event {self.event_kind}>"
