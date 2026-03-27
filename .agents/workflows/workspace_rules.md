---
description: Coding standards and architectural rules for the Facial Recognition Service
---

# Workspace Rules: Facial Recognition Service

These rules govern the development and maintenance of this repository. They are derived from the project's established patterns and the developer's preferences.

## 1. Core Technology Stack
- **Framework**: Flask (API) + SQLAlchemy (ORM).
- **Computer Vision**: `face_recognition` (dlib) and OpenCV.
- **Database**: PostgreSQL with `pgvector` for 128-D vector similarity.
- **Concurrency**: All CPU-heavy face processing MUST be wrapped in the `_face_processing_lock` to prevent segfaults in dlib's C++ core.

## 2. Coding Standards
- **Logging**: Use `logging.getLogger(__name__)`. Never use `print()` for server telemetry.
- **Documentation**: Every module and function MUST have a descriptive docstring explaining its purpose and logic.
- **Arguments**: Use `argparse` for CLI options (e.g., `--loglevel`, `--debug-webserver`).

## 3. Biometric & Business Logic
- **Quality Gates**: Registrations require a minimum face resolution of 100x100 and a sharpness (Variance of Laplacian) > 75.0.
- **Auto-healing**: During recognition, if a match is found with an image >10% sharper than the stored baseline (and >100.0 absolute score), automatically update the database record.
- **Deduplication**: Store face images using the SHA256 hash of the binary data as the filename.

## 4. Maintenance
- **Migrations**: Always use `flask db upgrade` via Flask-Migrate for schema changes.
- **Cleanup**: User deletion must trigger a cascade delete in the DB and remove physical image files from the `face_storage` directory.
