"""
B2B E-commerce Platform - Complete Backend (PostgreSQL Migration)
"""

from fastapi import FastAPI, Depends, HTTPException, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import uvicorn
from enum import Enum
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration - PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://b2b_ecommerce_platform_user:XrQVSwJxcihJt8eqfx0y2iFjuY4L3haT@dpg-d2gtsr2dbo4c73ahnlt0-a.singapore-postgres.render.com/b2b_ecommerce_platform"
)

# Add SSL mode for Render PostgreSQL
if "sslmode" not in DATABASE_URL:
    DATABASE_URL += "?sslmode=prefer"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=10,
    max_overflow=20,
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "development-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    SUPPLIER = "supplier"
    BUYER = "buyer"

class OrderStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

# Database Models (Updated for PostgreSQL)
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    company_name = Column(String(255))
    role = Column(String(50), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    products = relationship("Product", back_populates="supplier")
    orders_as_buyer = relationship("Order", foreign_keys="[Order.buyer_id]", back_populates="buyer")
    cart_items = relationship("CartItem", back_populates="user")

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    min_order_quantity = Column(Integer, default=1)
    category = Column(String(100), index=True)
    sku = Column(String(100), unique=True, index=True)
    supplier_id = Column(Integer, ForeignKey("users.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    supplier = relationship("User", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")
    cart_items = relationship("CartItem", back_populates="product")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(100), unique=True, index=True)
    buyer_id = Column(Integer, ForeignKey("users.id"))
    status = Column(String(50), default=OrderStatus.PENDING)
    total_amount = Column(Float, nullable=False)
    shipping_address = Column(Text)
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    buyer = relationship("User", foreign_keys=[buyer_id], back_populates="orders_as_buyer")
    order_items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")

class CartItem(Base):
    __tablename__ = "cart_items"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, nullable=False)
    added_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="cart_items")
    product = relationship("Product", back_populates="cart_items")

# Pydantic Models (unchanged)
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    company_name: Optional[str] = None
    role: UserRole
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    company_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    stock_quantity: int = 0
    min_order_quantity: int = 1
    category: Optional[str] = None
    sku: str

class ProductResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    price: float
    stock_quantity: int
    min_order_quantity: int
    category: Optional[str]
    sku: str
    supplier_id: int
    supplier_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class CartItemCreate(BaseModel):
    product_id: int
    quantity: int

class CartItemResponse(BaseModel):
    id: int
    product_id: int
    product_name: str
    product_price: float
    quantity: int
    total_price: float
    
    class Config:
        from_attributes = True

class OrderCreate(BaseModel):
    shipping_address: str
    notes: Optional[str] = None

class OrderResponse(BaseModel):
    id: int
    order_number: str
    buyer_id: int
    status: str
    total_amount: float
    shipping_address: str
    notes: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

# FastAPI App
app = FastAPI(
    title="B2B E-commerce Platform",
    description="A comprehensive B2B e-commerce platform with PostgreSQL backend",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory and mount
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database initialization
def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication utilities (unchanged)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), 
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Routes (unchanged but with improved error handling)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend application"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure index.html is in the project root.</p>",
            status_code=404
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint with PostgreSQL database test"""
    try:
        db = SessionLocal()
        # Test database connection
        result = db.execute(text("SELECT 1")).fetchone()
        
        if result:
            user_count = db.query(User).count()
            product_count = db.query(Product).count()
            order_count = db.query(Order).count()
            db.close()
            
            return {
                "status": "healthy",
                "message": "B2B E-commerce Platform is running with PostgreSQL",
                "version": "2.0.0",
                "timestamp": datetime.utcnow(),
                "database": {
                    "type": "PostgreSQL",
                    "status": "connected",
                    "host": "dpg-d2gtsr2dbo4c73ahn1t0-a.singapore-postgres.render.com"
                },
                "statistics": {
                    "users": user_count,
                    "products": product_count,
                    "orders": order_count
                }
            }
        else:
            db.close()
            raise Exception("Database query failed")
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "message": "Service issues detected",
            "error": str(e),
            "timestamp": datetime.utcnow(),
            "database": {
                "type": "PostgreSQL",
                "status": "disconnected"
            }
        }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint for API verification"""
    return {
        "message": "API is working correctly with PostgreSQL",
        "timestamp": datetime.utcnow(),
        "database": "PostgreSQL on Render",
        "endpoints": {
            "frontend": "/",
            "api_docs": "/api/docs",
            "health": "/api/health",
            "auth": "/api/auth/login",
            "products": "/api/products",
            "seed": "/api/test/seed-database"
        }
    }

# Authentication Routes
@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        db_user = db.query(User).filter(
            (User.email == user.email) | (User.username == user.username)
        ).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email or username already registered")
        
        hashed_password = get_password_hash(user.password)
        db_user = User(
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            company_name=user.company_name,
            role=user.role,
            hashed_password=hashed_password
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        logger.info(f"New user registered: {user.username} ({user.role})")
        return db_user
    except Exception as e:
        db.rollback()
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login", response_model=Token)
async def login_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Authenticate user and return JWT token"""
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# Product Routes
@app.get("/api/products", response_model=List[ProductResponse])
async def list_products(
    skip: int = 0, 
    limit: int = 100, 
    category: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all active products with optional filtering"""
    query = db.query(Product).filter(Product.is_active == True)
    
    if category:
        query = query.filter(Product.category == category)
    
    if search:
        query = query.filter(Product.name.ilike(f"%{search}%"))
    
    products = query.offset(skip).limit(limit).all()
    
    # Add supplier name to each product
    for product in products:
        supplier = db.query(User).filter(User.id == product.supplier_id).first()
        product.supplier_name = supplier.company_name if supplier else "Unknown"
    
    return products

@app.post("/api/products", response_model=ProductResponse)
async def create_product(
    product: ProductCreate, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    """Create a new product (Suppliers and Admins only)"""
    if current_user.role not in [UserRole.ADMIN, UserRole.SUPPLIER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only suppliers and admins can create products"
        )
    
    try:
        existing_product = db.query(Product).filter(Product.sku == product.sku).first()
        if existing_product:
            raise HTTPException(status_code=400, detail="Product with this SKU already exists")
        
        db_product = Product(
            **product.dict(),
            supplier_id=current_user.id
        )
        db.add(db_product)
        db.commit()
        db.refresh(db_product)
        return db_product
    except Exception as e:
        db.rollback()
        logger.error(f"Product creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Product creation failed")

# Cart Routes
@app.post("/api/cart/add")
async def add_to_cart(
    item: CartItemCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add item to cart"""
    if current_user.role != UserRole.BUYER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only buyers can add items to cart"
        )
    
    try:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        existing_item = db.query(CartItem).filter(
            CartItem.user_id == current_user.id,
            CartItem.product_id == item.product_id
        ).first()
        
        if existing_item:
            existing_item.quantity += item.quantity
        else:
            cart_item = CartItem(
                user_id=current_user.id,
                product_id=item.product_id,
                quantity=item.quantity
            )
            db.add(cart_item)
        
        db.commit()
        return {"message": "Item added to cart successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Add to cart failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add item to cart")

@app.get("/api/cart", response_model=List[CartItemResponse])
async def get_cart(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's cart items"""
    cart_items = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
    
    result = []
    for item in cart_items:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if product:
            result.append({
                "id": item.id,
                "product_id": item.product_id,
                "product_name": product.name,
                "product_price": product.price,
                "quantity": item.quantity,
                "total_price": product.price * item.quantity
            })
    
    return result

@app.delete("/api/cart/{item_id}")
async def remove_from_cart(
    item_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Remove item from cart"""
    try:
        cart_item = db.query(CartItem).filter(
            CartItem.id == item_id,
            CartItem.user_id == current_user.id
        ).first()
        
        if not cart_item:
            raise HTTPException(status_code=404, detail="Cart item not found")
        
        db.delete(cart_item)
        db.commit()
        return {"message": "Item removed from cart"}
    except Exception as e:
        db.rollback()
        logger.error(f"Remove from cart failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to remove item from cart")

@app.delete("/api/cart/clear")
async def clear_cart(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear user's cart"""
    try:
        deleted_count = db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()
        db.commit()
        return {"message": f"Cart cleared. {deleted_count} items removed."}
    except Exception as e:
        db.rollback()
        logger.error(f"Clear cart failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cart")

# Order Routes
@app.post("/api/orders", response_model=OrderResponse)
async def create_order(
    order: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create order from cart items"""
    if current_user.role != UserRole.BUYER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only buyers can create orders"
        )
    
    try:
        cart_items = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
        if not cart_items:
            raise HTTPException(status_code=400, detail="Cart is empty")
        
        total_amount = 0
        order_items_data = []
        
        for cart_item in cart_items:
            product = db.query(Product).filter(Product.id == cart_item.product_id).first()
            if not product:
                continue
            
            item_total = product.price * cart_item.quantity
            total_amount += item_total
            
            order_items_data.append({
                "product_id": product.id,
                "quantity": cart_item.quantity,
                "price": product.price
            })
        
        # Generate order number
        order_count = db.query(Order).count()
        order_number = f"ORD-{order_count + 1:06d}"
        
        # Create order
        db_order = Order(
            order_number=order_number,
            buyer_id=current_user.id,
            total_amount=total_amount,
            shipping_address=order.shipping_address,
            notes=order.notes
        )
        db.add(db_order)
        db.commit()
        db.refresh(db_order)
        
        # Create order items
        for item_data in order_items_data:
            order_item = OrderItem(
                order_id=db_order.id,
                **item_data
            )
            db.add(order_item)
        
        # Clear cart
        db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()
        
        db.commit()
        logger.info(f"Order created: {order_number} by {current_user.username}")
        
        return db_order
    except Exception as e:
        db.rollback()
        logger.error(f"Order creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create order")

@app.get("/api/orders", response_model=List[OrderResponse])
async def list_orders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's orders"""
    if current_user.role == UserRole.BUYER:
        orders = db.query(Order).filter(Order.buyer_id == current_user.id).all()
    elif current_user.role == UserRole.ADMIN:
        orders = db.query(Order).all()
    else:
        # Suppliers see orders for their products
        orders = db.query(Order).join(OrderItem).join(Product).filter(
            Product.supplier_id == current_user.id
        ).distinct().all()
    
    return orders

# Testing Routes
@app.post("/api/test/seed-database")
async def seed_database(db: Session = Depends(get_db)):
    """Seed database with test data"""
    
    try:
        # Create test users
        test_users = [
            {
                "email": "admin@test.com",
                "username": "admin",
                "full_name": "System Admin",
                "company_name": "Platform Admin",
                "role": UserRole.ADMIN,
                "password": "admin123"
            },
            {
                "email": "supplier1@test.com",
                "username": "supplier1",
                "full_name": "John Supplier",
                "company_name": "Tech Supplies Co.",
                "role": UserRole.SUPPLIER,
                "password": "supplier123"
            },
            {
                "email": "buyer1@test.com",
                "username": "buyer1",
                "full_name": "Jane Buyer",
                "company_name": "ABC Corporation",
                "role": UserRole.BUYER,
                "password": "buyer123"
            }
        ]
        
        created_users = {}
        for user_data in test_users:
            existing_user = db.query(User).filter(User.email == user_data["email"]).first()
            if not existing_user:
                hashed_password = get_password_hash(user_data["password"])
                user = User(
                    email=user_data["email"],
                    username=user_data["username"],
                    full_name=user_data["full_name"],
                    company_name=user_data["company_name"],
                    role=user_data["role"],
                    hashed_password=hashed_password
                )
                db.add(user)
                db.commit()
                db.refresh(user)
                created_users[user_data["role"]] = user
            else:
                created_users[user_data["role"]] = existing_user
        
        # Get supplier for products
        supplier = created_users.get(UserRole.SUPPLIER) or db.query(User).filter(User.role == UserRole.SUPPLIER).first()
        
        # Create test products
        test_products = [
            {
                "name": "Laptop Dell XPS 13",
                "description": "High-performance ultrabook for business professionals",
                "price": 1299.99,
                "stock_quantity": 50,
                "min_order_quantity": 1,
                "category": "Electronics",
                "sku": "DELL-XPS13-001"
            },
            {
                "name": "Office Chair Ergonomic",
                "description": "Comfortable ergonomic office chair with lumbar support",
                "price": 299.99,
                "stock_quantity": 30,
                "min_order_quantity": 5,
                "category": "Furniture",
                "sku": "CHAIR-ERG-001"
            },
            {
                "name": "Wireless Mouse Logitech",
                "description": "Wireless optical mouse with USB receiver",
                "price": 29.99,
                "stock_quantity": 100,
                "min_order_quantity": 10,
                "category": "Electronics",
                "sku": "MOUSE-LOG-001"
            }
        ]
        
        products_created = 0
        for product_data in test_products:
            existing_product = db.query(Product).filter(Product.sku == product_data["sku"]).first()
            if not existing_product:
                product = Product(
                    **product_data,
                    supplier_id=supplier.id
                )
                db.add(product)
                products_created += 1
        
        db.commit()
        
        return {
            "message": "Database seeded successfully with test data",
            "database": "PostgreSQL on Render",
            "users_created": len(created_users),
            "products_created": products_created,
            "test_credentials": {
                "admin": "admin / admin123",
                "supplier": "supplier1 / supplier123", 
                "buyer": "buyer1 / buyer123"
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Database seeding failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to seed database")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )