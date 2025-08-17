"""
Application configuration using Pydantic Settings
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Project Information
    PROJECT_NAME: str = "B2B E-commerce Platform"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "A comprehensive B2B e-commerce platform for testing"
    
    # Environment
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security
    SECRET_KEY: str = Field(
        default="development-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./b2b_ecommerce.db",
        env="DATABASE_URL"
    )
    
    # CORS
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"],
        env="ALLOWED_HOSTS"
    )
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Testing
    TESTING: bool = Field(default=False, env="TESTING")
    TEST_DATABASE_URL: str = Field(
        default="sqlite:///./test_b2b_ecommerce.db",
        env="TEST_DATABASE_URL"
    )
    
    # Business Logic
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL"""
        if self.TESTING:
            return self.TEST_DATABASE_URL
        return self.DATABASE_URL
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.TESTING or self.ENVIRONMENT.lower() == "testing"


# Create global settings instance
settings = Settings()