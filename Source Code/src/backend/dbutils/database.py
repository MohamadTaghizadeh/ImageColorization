from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.config_handler import config
import os
from loguru import logger


def _fk_pragma_on_connect(dbapi_con, con_record):
    """Enable foreign key support for SQLite"""
    dbapi_con.execute("pragma foreign_keys=ON")


# Create database directory if it doesn't exist
os.makedirs(config.get("DB_DIR", "db"), exist_ok=True)

# Create database engine based on connection string
connection_string = config.get("DB_CONNECTION", "sqlite:///db/videototxt.db")

if "sqlite" in connection_string:
    engine = create_engine(
        connection_string,
        connect_args={"check_same_thread": False},
        pool_recycle=3600,
    )
    # Enable foreign key support for SQLite
    from sqlalchemy import event
    event.listen(engine, "connect", _fk_pragma_on_connect)
    logger.info(f"Created SQLite database engine: {connection_string}")
elif "mysql" in connection_string:
    engine = create_engine(connection_string, pool_recycle=3600)
    logger.info(f"Created MySQL database engine: {connection_string}")
elif "oracle" in connection_string:
    engine = create_engine(connection_string, pool_recycle=3600)
    logger.info(f"Created Oracle database engine: {connection_string}")
else:
    logger.warning(f"Unknown database type in connection string: {connection_string}")
    engine = create_engine(connection_string, pool_recycle=3600)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()


def init_db():
    """
    Initialize the database by creating all tables
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
