"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from .config import settings

# Create database engine
engine = create_engine(
    settings.database_url_sync,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url_sync else {},
    pool_pre_ping=True,
    echo=settings.is_development and not settings.is_testing
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)


def reset_database() -> None:
    """Reset database by dropping and recreating all tables"""
    drop_tables()
    create_tables()