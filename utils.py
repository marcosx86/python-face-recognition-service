"""
Computer Vision utilities for the Facial Recognition Service.
Handles image processing, face detection, and biometric embedding extraction
using dlib (via face_recognition) and OpenCV.
"""
import face_recognition
import cv2
import numpy as np

def extract_face_encoding(image_stream, registration_mode=False):
    """
    Detects a face in the provided image stream and extracts its 128-D embedding.
    
    Args:
        image_stream (file-like): The image data stream (e.g., Flask FileStorage).
        registration_mode (bool): If True, applies strict biometric quality requirements
                                  including minimum face resolution and sharpness/focus thresholds.
                                  
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The 128-dimensional face encoding vector.
            - float: The calculated focus/sharpness score (Variance of Laplacian).
            
    Raises:
        ValueError: If no faces are found, multiple faces are found, or if 
                    the image fails biometric quality checks during registration.
    """
    # Load the image using the face_recognition library
    # It dynamically handles numpy arrays and file-like objects (like those passed by Flask)
    image = face_recognition.load_image_file(image_stream)
    
    # 1. Detect where faces are located in the image
    # We use the default HOG (Histogram of Oriented Gradients) model because it's fast on the CPU
    face_locations = face_recognition.face_locations(image, model="hog")
    
    if len(face_locations) == 0:
        raise ValueError("No faces were found in the uploaded image.")
    top, right, bottom, left = face_locations[0]
    face_width = right - left
    face_height = bottom - top
    
    # Extract the cropped region of just the face from the original RGB array
    face_image = image[top:bottom, left:right]
    # Convert to grayscale for OpenCV
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    # Calculate the variance of the laplacian; higher is sharper
    blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    if registration_mode:
        # Biometric Check A: Face Resolution
        # Standard requirement for matching is often > 100px.
        if face_width < 100 or face_height < 100:
            raise ValueError(f"Face resolution is too low ({face_width}x{face_height} pixels). For registration, the face must be at least 100x100 pixels in the frame.")
            
        # Biometric Check B: Focus/Blurriness (Variance of Laplacian)
        # Blur Threshold: 100 is a standard baseline, but you may need to tune this depending on the environment.
        if blur_score < 75.0:
            raise ValueError(f"Face appears to be too blurry or out of focus (Focus score: {blur_score:.1f}). Please capture a sharper image.")

    # 2. Extract the 128-dimensional embedding for the recognized face
    face_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    
    return face_encodings[0], float(blur_score)
