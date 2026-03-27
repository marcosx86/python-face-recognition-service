"""
Database Initialization Utilities for the Facial Recognition Service.
Provides a standalone script to manually bootstrap the PostgreSQL database,
ensuring required extensions (e.g., pgvector) and tables are installed.
"""
import logging
import argparse
from app import app
from models import db
from sqlalchemy import text

logger = logging.getLogger(__name__)

def setup_database():
    """
    Creates the required PostgreSQL extensions and missing tables.
    Must be run before starting the Flask server.
    """
    logger.debug("Executing setup_database().")
    with app.app_context():
        logger.info("Checking for pgvector extension...")
        try:
            # We attempt to create the pgvector extension if it is entirely missing.
            # This requires superuser privileges inside PostgreSQL. If you already ran
            # 'CREATE EXTENSION vector;' externally, this step silently passes or warns.
            db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            db.session.commit()
            logger.info("pgvector extension ready.")
        except Exception as e:
            db.session.rollback()
            logger.warning("Could not auto-create standard extension 'vector'.")
            logger.warning("Ensure your DB user has permissions or run 'CREATE EXTENSION vector;' manually.")
            logger.warning(f"Error detail: {e}")
            
        logger.info("Creating table structures (if missing)...")
        db.create_all()
        
        logger.info("Ensuring HNSW index exists for optimized vector search...")
        try:
            # Create HNSW index for L2 distance if it doesn't exist
            # Note: Postgres 15+ supports 'IF NOT EXISTS' for indexes directly.
            # For broader compatibility, we just attempt it and catch the 'already exists' error if needed,
            # but 'CREATE INDEX IF NOT EXISTS' is standard in modern Postgres/pgvector setups.
            db.session.execute(text('CREATE INDEX IF NOT EXISTS idx_face_encodings_vector ON face_encodings USING hnsw (encoding vector_l2_ops)'))
            db.session.commit()
            logger.info("HNSW vector index is ready.")
        except Exception as e:
            db.session.rollback()
            logger.warning(f"Could not create HNSW index: {e}. Performance may be degraded for large datasets.")
            
        logger.info("Done! Database tables and indexes are ready.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize Database for Facial Recognition Service")
    parser.add_argument('--loglevel', default='INFO', help='Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {args.loglevel}')
    
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    setup_database()
