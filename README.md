# Python Facial Recognition Service

A high-performance REST API for registering and identifying faces using machine learning and vector similarity search.

This service combines the accuracy of `dlib`-powered `face_recognition` with the blazing speed of PostgreSQL's `pgvector` extension. It allows you to store human faces as 128-dimensional mathematical embeddings and match against them with scale and precision.

## Core Features

- **Biometric Registration**: Automatically detects the primary face in an uploaded image, runs strict resolution and focus (laplacian variance) checks, and securely hashes the face to a `pgvector` embedding.
- **Lightning Fast Matching**: Performs near-instantaneous Euclidean L2-distance queries inside PostgreSQL to find the closest matching user profile within a secure confidence threshold.
- **Continuous Auto-Healing**: If a user is recognized using an image that is significantly crisper or higher-resolution than the one they initially registered with, the API silently upgrades their database profile and saves the new high-quality reference image.
- **Image Deduplication**: Original images are stored to disk securely with their SHA256 hashes used as filenames, naturally deduplicating redundant uploads.
- **Flame Graph Profiling**: Every `/sync` response includes a `flame_graph` field with millisecond-level timings for each internal step (image loading, face detection, encoding, DB query, commit), enabling instant performance visibility.
- **Container Ready**: Fully Dockerized and configurable via OS environment variables.
- **CORS Enabled**: Accepts cross-origin requests from any domain, suitable for web and mobile clients.

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
*Note: The Docker container automatically applies any pending Alembic database schemas on startup before launching the Waitress WSGI server.*

### Running Locally

1. Create a virtual environment and load requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # Or `.\\.venv\\Scripts\\activate` on Windows
pip install -r requirements.txt
```

2. Run the application (which will also apply Alembic migrations automatically):
```bash
python app.py
# For development with auto-reload:
python app.py --loglevel DEBUG --debug-webserver
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

### 3. All-in-One Web Sync
`POST /api/v1/sync`
- Accepts `multipart/form-data`.
- **Fields:**
  - `image`: The high-rate frame (1-2 FPS).
  - `identifier`: (Optional) The name to register if the person is not recognized.
  - `max_faces`: (Optional, default `0`) Maximum number of faces to process per frame. Faces are prioritized by size (largest first). `0` means unlimited.
- **Response:** Returns a list of detected faces (`identifier`, `box`, `confidence`), total `server_process_time_ms`, and a `flame_graph` array with per-step timing breakdowns.

**Example `flame_graph` response field:**
```json
"flame_graph": [
  { "label": "load_image_file",  "duration_ms": 26.04 },
  { "label": "face_locations",   "duration_ms": 16.74 },
  { "label": "face_encodings",   "duration_ms": 563.70 },
  { "label": "blur_check_loop",  "duration_ms": 0.55  },
  { "label": "similarity_search","duration_ms": 21.07 },
  { "label": "log_detection",    "duration_ms": 0.13  },
  { "label": "commit",           "duration_ms": 12.50 }
]
```

### 4. List Registered Identifiers
`GET /api/v1/users`
- Returns a JSON array of all registered text identifiers.

### 5. Fetch the User's Best Face
`GET /api/v1/users/<identifier>/face`
- Returns the exact original image file (highest quality capture saved) for the requested identifier.

### 6. Webhook Management
`POST /api/v1/webhooks` – Register a webhook for a user/event.
`GET /api/v1/webhooks` – List all webhooks (filterable by `?identifier=`).
`DELETE /api/v1/webhooks/<id>` – Remove a webhook.

## Architecture & Technology Stack

- **Web Framework:** Flask + Waitress (production WSGI) / Werkzeug (dev)
- **Database:** PostgreSQL + psycopg2 + pgvector
- **ORM:** SQLAlchemy + Flask-Migrate (Alembic)
- **Computer Vision:** `dlib`, `face_recognition`, OpenCV (`cv2`)
- **Cross-Origin:** Flask-CORS (allow all)
