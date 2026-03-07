# Facial Recognition Service Architecture Plan

This document outlines the proposed architecture, components, and configuration for a Python-based facial recognition service using Flask.

## 1. System Layers

### API Layer
*   **Framework:** Flask
*   **Purpose:** Handles incoming HTTP requests, route definitions, input validation, and JSON serialization.
*   **Responsibilities:** Receiving images from clients, returning match results or error states.

### Processing Layer (Computer Vision)
*   **Core Libraries:** `face_recognition` (built on `dlib`) and `opencv-python-headless`.
*   **Purpose:** Handles the extraction of facial embeddings (encodings) from uploaded images and computing the distances/matches between faces.
*   **Responsibilities:** Detecting faces in images, generating 128-dimensional encodings, and numerical comparison.

### Data Storage Layer
*   **Faces/Embeddings Database:** PostgreSQL with the `pgvector` extension.
    *   *Why PostgreSQL?* PostgreSQL's `pgvector` extension is specifically built for storing vector embeddings (like 128-D face encodings) and performing extremely fast similarity searches (using indexes like HNSW or IVFFlat). MariaDB has vector capabilities recently but PostgreSQL is the industry standard for this exact use case.
*   **Image Storage:** Local file system for storing original reference images.
*   **Purpose:** Persisting registered users and their corresponding numerical face encodings for fast lookups.

## 2. Core Components

*   **Registration Module (`/api/v1/register`):** 
    *   Accepts an image file and a user identifier.
    *   Validates the image contains exactly one face.
    *   Extracts the face encoding and saves it to the database alongside the user ID.
*   **Recognition Module (`/api/v1/recognize`):** 
    *   Accepts an image.
    *   Extracts face encodings.
    *   Compares extracted encodings against the database to find the closest match within a specified tolerance.
*   **Image Processing Utilities (`utils.py`):** 
    *   Encapsulates the complex CV logic to keep the Flask application routes clean and testable.
*   **Database Models (`models.py`):** 
    *   Defines the schema, e.g., a `User` table and potentially a one-to-many `FaceEncoding` table storing pickled or JSON-serialized numpy arrays.

## 3. Configuration & Environment Requirements

### System Dependencies
*   C++ Compiler (e.g., Visual Studio C++ Build Tools on Windows, `build-essential` on Linux)
*   `CMake` (required to compile the underlying `dlib` library)
*   **CUDA 12.6 & cuDNN:** Required to run the face detection/encoding models on the GPU.
*   **Note on Python 3.12 & CUDA 12.6 (`dlib`):** While there are community pre-built `.whl` files for Python 3.12, they almost exclusively target CPU-only usage. To utilize **CUDA 12.6** for hardware acceleration, you will need to compile `dlib` from source. 
    *   *Steps for compilation:* Install Visual Studio C++ Build Tools, CUDA Toolkit 12.6, and cuDNN. Then `pip install cmake`, clone the `dlib` repository, and run `python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA`.

### Python Package Dependencies (`requirements.txt`)
*   `Flask` (Web framework)
*   `face_recognition` (High-level face recognition API)
*   `opencv-python-headless` (Image manipulation)
*   `numpy` (Required for mathematical distance operations on encodings)
*   `Flask-SQLAlchemy` & `psycopg2` & `pgvector` (Database connection and vector data types)

## User Review Required

Please review this updated architecture breakdown. If you approve of the PostgreSQL + `pgvector` choice and the strategy for compiling `dlib` with CUDA 12.6, let me know, and we can proceed to the next iteration: setting up the actual project code and folder structure!
