"""
Database Initialization Utilities for the Facial Recognition Service.
Provides a standalone script to manually bootstrap the PostgreSQL database,
ensuring required extensions (e.g., pgvector) and tables are installed.
"""
from app import app
from models import db
from sqlalchemy import text

def setup_database():
    """
    Creates the required PostgreSQL extensions and missing tables.
    Must be run before starting the Flask server.
    """
    with app.app_context():
        print("Checking for pgvector extension...")
        try:
            # We attempt to create the pgvector extension if it is entirely missing.
            # This requires superuser privileges inside PostgreSQL. If you already ran
            # 'CREATE EXTENSION vector;' externally, this step silently passes or warns.
            db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            db.session.commit()
            print("pgvector extension ready.")
        except Exception as e:
            db.session.rollback()
            print(f"Notice: Could not auto-create standard extension 'vector'.")
            print(f"Ensure your DB user has permissions or run 'CREATE EXTENSION vector;' manually.")
            print(f"Error detail: {e}")
            
        print("Creating table structures (if missing)...")
        db.create_all()
        print("Done! Database tables are ready.")

if __name__ == '__main__':
    setup_database()
