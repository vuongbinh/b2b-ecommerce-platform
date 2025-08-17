"""
B2B E-commerce Platform - Main Entry Point
Enhanced version with Poetry dependency management
"""

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.auth import router as auth_router
from src.api.products import router as products_router
from src.api.cart import router as cart_router
from src.api.orders import router as orders_router
from src.api.admin import router as admin_router
from src.core.config import settings
from src.core.database import engine, Base
from src.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting B2B E-commerce Platform...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down B2B E-commerce Platform...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A comprehensive B2B e-commerce platform designed for Playwright testing",
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(products_router, prefix="/api/products", tags=["Products"])
app.include_router(cart_router, prefix="/api/cart", tags=["Shopping Cart"])
app.include_router(orders_router, prefix="/api/orders", tags=["Orders"])
app.include_router(admin_router, prefix="/api/admin", tags=["Administration"])


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend application"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <h1>B2B E-commerce Platform</h1>
            <p>Frontend not found. Please ensure index.html is in the project root.</p>
            <p><a href="/api/docs">View API Documentation</a></p>
            """,
            status_code=404
        )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    from src.core.database import SessionLocal
    
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
            "version": settings.VERSION,
            "database": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )