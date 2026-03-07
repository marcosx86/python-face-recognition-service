# Python Facial Recognition Service

A high-performance REST API for registering and identifying faces using machine learning and vector similarity search.

This service combines the accuracy of `dlib`-powered `face_recognition` with the blazing speed of PostgreSQL's `pgvector` extension. It allows you to store human faces as 128-dimensional mathematical embeddings and match against them with scale and precision.

## Core Features

- **Biometric Registration**: Automatically detects the primary face in an uploaded image, runs strict resolution and focus (laplacian variance) checks, and securely hashes the face to a `pgvector` embedding.
- **Lightning Fast Matching**: Performs near-instantaneous Euclidean L2-distance queries inside PostgreSQL to find the closest matching user profile within a secure confidence threshold.
- **Continuous Auto-Healing**: If a user is recognized using an image that is significantly crisper or higher-resolution than the one they initially registered with, the API silently upgrades their database profile and saves the new high-quality reference image.
- **Image Deduplication**: Original images are stored to disk securely with their SHA256 hashes used as filenames, naturally deduplicating redundant uploads.
- **Container Ready**: Fully Dockerized and configurable via OS environment variables.

## Getting Started

### Prerequisites

- Docker OR a Python 3.12+ environment with standard build tools (`cmake`, `build-essential`)
- A running PostgreSQL Server with the [pgvector extension](https://github.com/pgvector/pgvector) installed.

### Configuration

Database connections and other options are handled via environment variables (or fallbacks in `config.py`):

- `DB_HOST`: Hostname of the Postgres server (Default: `192.168.88.71`)
- `DB_PORT`: Postgres port (Default: `5434`)
- `DB_USER`: Postgres User (Default: `facialrecognition`)
- `DB_PASSWORD`: Postgres Password (Default: `123Mud4r!`)
- `DB_NAME`: Database Name (Default: `facialrecognition`)

### Running with Docker (Recommended)

Build the image:
```bash
docker build -t facial-recog-service .
```

Run the container (passing credentials through `-e`):
```bash
docker run -d -p 5000:5000 \
  -e DB_HOST="your-db-host" \
  -e DB_USER="your-db-user" \
  -e DB_PASSWORD="YourSecurePassword" \
  facial-recog-service
```
*Note: The Docker container automatically applies any pending Alembic database schemas on startup before launching the Gunicorn server.*

### Running Locally

1. Create a virtual environment and load requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # Or `.\.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

2. Run the application (which will also apply Alembic migrations automatically):
```bash
python app.py
```

## API Reference

### 1. Register a Face
`POST /api/v1/register`
- Accepts `multipart/form-data`.
- **Fields:**
  - `image`: The `.jpg` or `.png` picture.
  - `identifier`: A unique string ID (e.g., employee number or email).

### 2. Recognize a Face
`POST /api/v1/recognize`
- Accepts `multipart/form-data`.
- **Fields:**
  - `image`: The face you want to identify.
- **Response:** Returns the closest `identifier`, the confidence distance, and whether the API auto-updated their baseline profile with this improved picture.

### 3. List Registered Identifiers
`GET /api/v1/users`
- Returns a JSON array of all registered text identifiers.

### 4. Fetch the User's Best Face
`GET /api/v1/users/<identifier>/face`
- Returns the exact original image file (highest quality capture saved) for the requested identifier.

## Architecture & Technology Stack

- **Web Framework:** Flask / Gunicorn
- **Database:** PostgreSQL + psycopg2 + pgvector
- **ORM:** SQLAlchemy + Flask-Migrate (Alembic)
- **Computer Vision:** `dlib`, `face_recognition`, OpenCV (`cv2`)
