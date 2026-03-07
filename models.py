"""
SQLAlchemy database models for the Facial Recognition Service.
Defines the User schema and their associated FaceEncodings utilizing
PostgreSQL's pgvector extension for efficient similarity matching.
"""
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector

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
    
    # One-to-many relationship: A user can have multiple face encodings (e.g., different profiles/lighting)
    encodings = db.relationship('FaceEncoding', backref='user', lazy=True, cascade="all, delete-orphan")
    
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
    
    def __repr__(self):
        return f"<FaceEncoding {self.id} for User {self.user_id}>"
