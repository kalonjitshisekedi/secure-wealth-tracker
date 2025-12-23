# Python Financial Dashboard - Complete Implementation Guide

## Overview
A secure, production-ready financial dashboard that reads from a private Google Drive spreadsheet, includes user authentication, and deploys on AWS.

---

## 1. Project Structure

```
financial-dashboard/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # User SQLAlchemy models
│   │   ├── financial.py        # Financial data models
│   │   └── schemas.py          # Pydantic schemas
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── routes.py           # Auth endpoints
│   │   ├── security.py         # JWT & password hashing
│   │   └── dependencies.py     # Auth dependencies
│   ├── sheets/
│   │   ├── __init__.py
│   │   ├── client.py           # Google Sheets client
│   │   ├── validator.py        # Data validation
│   │   └── cache.py            # Caching layer
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── routes.py           # Dashboard API endpoints
│   │   └── aggregations.py     # Financial calculations
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py       # Database connection
│   │   └── migrations/         # Alembic migrations
│   └── templates/              # Jinja2 templates (if using SSR)
├── tests/
│   ├── __init__.py
│   ├── test_auth.py
│   ├── test_sheets.py
│   └── test_dashboard.py
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── terraform/              # AWS infrastructure
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── scripts/
│       ├── deploy.sh
│       └── setup_secrets.sh
├── requirements.txt
├── requirements-dev.txt
├── alembic.ini
├── .env.example
└── README.md
```

---

## 2. Google Sheets Template Specification

### Sheet Name: `Financial_Tracker`

| Column | Header | Data Type | Format | Required | Validation Rules |
|--------|--------|-----------|--------|----------|------------------|
| A | `transaction_id` | String | UUID | Yes | Unique identifier, auto-generated |
| B | `date` | Date | YYYY-MM-DD | Yes | Valid date, not future for expenses |
| C | `category` | String | Text | Yes | From predefined list (see below) |
| D | `subcategory` | String | Text | No | Dependent on category |
| E | `description` | String | Text | Yes | Max 500 characters |
| F | `amount` | Number | Currency | Yes | Positive number, 2 decimal places |
| G | `type` | String | Text | Yes | "income" or "expense" |
| H | `account` | String | Text | Yes | From predefined list |
| I | `payment_method` | String | Text | No | "cash", "credit", "debit", "transfer" |
| J | `tags` | String | Text | No | Comma-separated tags |
| K | `recurring` | Boolean | TRUE/FALSE | No | Default FALSE |
| L | `recurring_frequency` | String | Text | No | "weekly", "monthly", "yearly" |
| M | `notes` | String | Text | No | Max 1000 characters |
| N | `created_at` | DateTime | ISO 8601 | Auto | Auto-populated timestamp |
| O | `updated_at` | DateTime | ISO 8601 | Auto | Auto-updated timestamp |

### Predefined Categories:

**Income:**
- Salary
- Freelance
- Investments
- Rental Income
- Dividends
- Refunds
- Gifts
- Other Income

**Expenses:**
- Housing (Rent/Mortgage, Utilities, Maintenance)
- Transportation (Fuel, Public Transit, Car Payment, Insurance)
- Food (Groceries, Dining Out, Delivery)
- Healthcare (Insurance, Medical, Pharmacy)
- Entertainment (Streaming, Events, Hobbies)
- Shopping (Clothing, Electronics, Home Goods)
- Personal (Gym, Self-care, Education)
- Financial (Savings, Investments, Debt Payment)
- Other Expenses

### Sheet: `Accounts`

| Column | Header | Data Type | Required |
|--------|--------|-----------|----------|
| A | `account_id` | String | Yes |
| B | `account_name` | String | Yes |
| C | `account_type` | String | Yes |
| D | `institution` | String | No |
| E | `current_balance` | Number | Yes |
| F | `currency` | String | Yes |

### Sheet: `Budgets`

| Column | Header | Data Type | Required |
|--------|--------|-----------|----------|
| A | `budget_id` | String | Yes |
| B | `category` | String | Yes |
| C | `monthly_limit` | Number | Yes |
| D | `year` | Number | Yes |
| E | `month` | Number | Yes |

---

## 3. Requirements Files

### requirements.txt
```txt
# Core Framework
fastapi==0.109.2
uvicorn[standard]==0.27.1
gunicorn==21.2.0

# Database
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
alembic==1.13.1
asyncpg==0.29.0

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9

# Google Sheets
google-auth==2.27.0
google-auth-oauthlib==1.2.0
google-api-python-client==2.116.0
gspread==6.0.2

# AWS
boto3==1.34.34
botocore==1.34.34

# Validation & Serialization
pydantic==2.6.1
pydantic-settings==2.1.0
email-validator==2.1.0.post1

# Caching
redis==5.0.1
cachetools==5.3.2

# Security
cryptography==42.0.2

# HTTP Client
httpx==0.26.0
aiohttp==3.9.3

# Utilities
python-dotenv==1.0.1
structlog==24.1.0
tenacity==8.2.3

# Date/Time
python-dateutil==2.8.2

# Templating (optional, for SSR)
jinja2==3.1.3
```

### requirements-dev.txt
```txt
-r requirements.txt

# Testing
pytest==8.0.0
pytest-asyncio==0.23.4
pytest-cov==4.1.0
httpx==0.26.0
factory-boy==3.3.0

# Code Quality
black==24.1.1
isort==5.13.2
flake8==7.0.0
mypy==1.8.0
pre-commit==3.6.0

# Security Scanning
bandit==1.7.7
safety==2.3.5
```

---

## 4. Core Application Code

### app/config.py
```python
"""
Configuration management using Pydantic Settings.
All secrets are loaded from environment variables or AWS Secrets Manager.
"""
from functools import lru_cache
from typing import Optional
import json

import boto3
from botocore.exceptions import ClientError
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Financial Dashboard"
    app_env: str = Field(default="development", env="APP_ENV")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")  # 5 minutes
    
    # JWT Authentication
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # Google Sheets
    google_sheets_spreadsheet_id: str = Field(..., env="GOOGLE_SHEETS_SPREADSHEET_ID")
    google_service_account_key: Optional[str] = Field(None, env="GOOGLE_SERVICE_ACCOUNT_KEY")
    
    # AWS
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_secrets_manager_secret_name: Optional[str] = Field(
        None, env="AWS_SECRETS_MANAGER_SECRET_NAME"
    )
    
    # Rate Limiting
    sheets_requests_per_minute: int = Field(default=60, env="SHEETS_REQUESTS_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("google_service_account_key", pre=True, always=True)
    def load_google_credentials(cls, v, values):
        """Load Google credentials from AWS Secrets Manager if not provided."""
        if v:
            return v
        
        secret_name = values.get("aws_secrets_manager_secret_name")
        if secret_name:
            try:
                return get_secret_from_aws(secret_name, "google_service_account_key")
            except Exception:
                pass
        return v


def get_secret_from_aws(secret_name: str, key: str) -> Optional[str]:
    """
    Retrieve a secret from AWS Secrets Manager.
    
    Args:
        secret_name: Name of the secret in AWS Secrets Manager
        key: Key within the secret JSON to retrieve
        
    Returns:
        The secret value or None if not found
    """
    client = boto3.client("secretsmanager")
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_data = json.loads(response["SecretString"])
        return secret_data.get(key)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "ResourceNotFoundException":
            raise ValueError(f"Secret {secret_name} not found")
        raise


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

### app/auth/security.py
```python
"""
Security utilities for authentication and authorization.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.config import get_settings

settings = get_settings()

# Password hashing context
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Increased rounds for better security
)


class TokenData(BaseModel):
    """Token payload data."""
    user_id: str
    email: str
    token_type: str = "access"
    exp: datetime


class TokenPair(BaseModel):
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate a password hash."""
    return pwd_context.hash(password)


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User's unique identifier
        email: User's email address
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )


def create_refresh_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT refresh token with longer expiration."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.refresh_token_expire_days
        )
    
    payload = {
        "sub": user_id,
        "email": email,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )


def create_token_pair(user_id: str, email: str) -> TokenPair:
    """Create both access and refresh tokens."""
    access_token = create_access_token(user_id, email)
    refresh_token = create_refresh_token(user_id, email)
    
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60
    )


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        
        return TokenData(
            user_id=payload.get("sub"),
            email=payload.get("email"),
            token_type=payload.get("type", "access"),
            exp=datetime.fromtimestamp(payload.get("exp"), tz=timezone.utc)
        )
    except JWTError:
        return None
```

### app/auth/dependencies.py
```python
"""
Authentication dependencies for FastAPI routes.
"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import decode_token, TokenData
from app.database.connection import get_db
from app.models.user import User

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user.
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials
    token_data = decode_token(token)
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if token_data.token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Fetch user from database
    user = await User.get_by_id(db, token_data.user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Dependency to ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user
```

### app/sheets/client.py
```python
"""
Google Sheets client with retry logic, rate limiting, and error handling.
"""
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import gspread
from google.oauth2.service_account import Credentials
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound
import structlog

from app.config import get_settings
from app.sheets.cache import SheetsCache
from app.sheets.validator import TransactionValidator, ValidationError

logger = structlog.get_logger()
settings = get_settings()


class RateLimiter:
    """Token bucket rate limiter for Google Sheets API."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
    
    def acquire(self) -> bool:
        """Acquire a token, waiting if necessary."""
        now = time.time()
        time_passed = now - self.last_update
        self.tokens = min(
            self.requests_per_minute,
            self.tokens + time_passed * (self.requests_per_minute / 60)
        )
        self.last_update = now
        
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        
        # Wait for token
        sleep_time = (1 - self.tokens) * (60 / self.requests_per_minute)
        time.sleep(sleep_time)
        self.tokens = 0
        return True


class GoogleSheetsClient:
    """
    Client for interacting with Google Sheets API.
    
    Features:
    - OAuth2 service account authentication
    - Automatic retry with exponential backoff
    - Rate limiting to respect API quotas
    - Caching layer for reducing API calls
    - Comprehensive error handling
    """
    
    def __init__(self):
        self.spreadsheet_id = settings.google_sheets_spreadsheet_id
        self.rate_limiter = RateLimiter(settings.sheets_requests_per_minute)
        self.cache = SheetsCache()
        self._client: Optional[gspread.Client] = None
        self._spreadsheet: Optional[gspread.Spreadsheet] = None
        
    def _get_credentials(self) -> Credentials:
        """Load Google service account credentials."""
        creds_json = settings.google_service_account_key
        
        if not creds_json:
            raise ValueError(
                "Google service account credentials not configured. "
                "Set GOOGLE_SERVICE_ACCOUNT_KEY environment variable."
            )
        
        try:
            creds_data = json.loads(creds_json)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in Google service account credentials")
        
        return Credentials.from_service_account_info(
            creds_data,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.readonly"
            ]
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIError)
    )
    def _initialize_client(self) -> gspread.Client:
        """Initialize the gspread client with retry logic."""
        if self._client is None:
            credentials = self._get_credentials()
            self._client = gspread.authorize(credentials)
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIError)
    )
    def _get_spreadsheet(self) -> gspread.Spreadsheet:
        """Get the spreadsheet with retry logic."""
        if self._spreadsheet is None:
            client = self._initialize_client()
            try:
                self._spreadsheet = client.open_by_key(self.spreadsheet_id)
            except SpreadsheetNotFound:
                logger.error(
                    "spreadsheet_not_found",
                    spreadsheet_id=self.spreadsheet_id
                )
                raise ValueError(
                    f"Spreadsheet not found. Verify the spreadsheet ID "
                    f"and that the service account has access."
                )
        return self._spreadsheet
    
    def get_transactions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch transactions from the spreadsheet.
        
        Args:
            start_date: Filter transactions from this date
            end_date: Filter transactions until this date
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of validated transaction dictionaries
        """
        cache_key = f"transactions:{start_date}:{end_date}"
        
        # Check cache first
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info("transactions_cache_hit", key=cache_key)
                return cached
        
        # Rate limit the API call
        self.rate_limiter.acquire()
        
        try:
            spreadsheet = self._get_spreadsheet()
            worksheet = spreadsheet.worksheet("Financial_Tracker")
            
            # Get all records
            records = worksheet.get_all_records()
            
            # Validate and transform
            validated_transactions = []
            validator = TransactionValidator()
            
            for i, record in enumerate(records, start=2):  # Start at row 2 (after header)
                try:
                    validated = validator.validate(record)
                    
                    # Apply date filters
                    if start_date and validated["date"] < start_date:
                        continue
                    if end_date and validated["date"] > end_date:
                        continue
                    
                    validated_transactions.append(validated)
                    
                except ValidationError as e:
                    logger.warning(
                        "transaction_validation_failed",
                        row=i,
                        error=str(e),
                        record=record
                    )
            
            # Cache the results
            self.cache.set(cache_key, validated_transactions)
            
            logger.info(
                "transactions_fetched",
                count=len(validated_transactions),
                total_rows=len(records)
            )
            
            return validated_transactions
            
        except WorksheetNotFound:
            logger.error("worksheet_not_found", name="Financial_Tracker")
            raise ValueError(
                "Financial_Tracker worksheet not found in spreadsheet"
            )
        except APIError as e:
            logger.error("sheets_api_error", error=str(e))
            raise
    
    def get_accounts(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch account information from the Accounts sheet."""
        cache_key = "accounts"
        
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        self.rate_limiter.acquire()
        
        try:
            spreadsheet = self._get_spreadsheet()
            worksheet = spreadsheet.worksheet("Accounts")
            records = worksheet.get_all_records()
            
            self.cache.set(cache_key, records)
            return records
            
        except WorksheetNotFound:
            logger.warning("accounts_worksheet_not_found")
            return []
    
    def get_budgets(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        """Fetch budget information from the Budgets sheet."""
        cache_key = f"budgets:{year}:{month}"
        
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        self.rate_limiter.acquire()
        
        try:
            spreadsheet = self._get_spreadsheet()
            worksheet = spreadsheet.worksheet("Budgets")
            records = worksheet.get_all_records()
            
            # Filter by year/month if specified
            if year or month:
                records = [
                    r for r in records
                    if (not year or r.get("year") == year) and
                       (not month or r.get("month") == month)
                ]
            
            self.cache.set(cache_key, records)
            return records
            
        except WorksheetNotFound:
            logger.warning("budgets_worksheet_not_found")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check connectivity to the spreadsheet.
        
        Returns:
            Health status dictionary
        """
        try:
            self.rate_limiter.acquire()
            spreadsheet = self._get_spreadsheet()
            
            return {
                "status": "healthy",
                "spreadsheet_title": spreadsheet.title,
                "last_updated": spreadsheet.lastUpdateTime,
                "worksheets": [ws.title for ws in spreadsheet.worksheets()]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
```

### app/sheets/validator.py
```python
"""
Data validation for spreadsheet data to prevent injection and corruption.
"""
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator, ValidationError as PydanticValidationError


class ValidationError(Exception):
    """Custom validation error with details."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


# Valid categories and types
VALID_INCOME_CATEGORIES = [
    "Salary", "Freelance", "Investments", "Rental Income",
    "Dividends", "Refunds", "Gifts", "Other Income"
]

VALID_EXPENSE_CATEGORIES = [
    "Housing", "Transportation", "Food", "Healthcare",
    "Entertainment", "Shopping", "Personal", "Financial", "Other Expenses"
]

VALID_ACCOUNTS = ["Checking", "Savings", "Credit Card", "Cash", "Investment"]
VALID_PAYMENT_METHODS = ["cash", "credit", "debit", "transfer", "check"]
VALID_RECURRING_FREQUENCIES = ["weekly", "monthly", "yearly"]


class TransactionSchema(BaseModel):
    """Pydantic schema for transaction validation."""
    
    transaction_id: str = Field(..., min_length=1, max_length=50)
    date: datetime
    category: str = Field(..., min_length=1, max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    amount: Decimal = Field(..., gt=0, decimal_places=2)
    type: str = Field(..., pattern="^(income|expense)$")
    account: str = Field(..., min_length=1, max_length=100)
    payment_method: Optional[str] = Field(None, max_length=50)
    tags: Optional[str] = Field(None, max_length=500)
    recurring: bool = False
    recurring_frequency: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=1000)
    
    @validator("category")
    def validate_category(cls, v, values):
        """Validate category based on transaction type."""
        trans_type = values.get("type")
        all_categories = VALID_INCOME_CATEGORIES + VALID_EXPENSE_CATEGORIES
        
        if v not in all_categories:
            # Allow custom categories but sanitize
            v = cls.sanitize_string(v)
        
        return v
    
    @validator("payment_method")
    def validate_payment_method(cls, v):
        """Validate payment method."""
        if v and v.lower() not in VALID_PAYMENT_METHODS:
            v = cls.sanitize_string(v)
        return v.lower() if v else None
    
    @validator("recurring_frequency")
    def validate_recurring_frequency(cls, v, values):
        """Validate recurring frequency is set when recurring is True."""
        if values.get("recurring") and not v:
            raise ValueError("recurring_frequency required when recurring is True")
        if v and v.lower() not in VALID_RECURRING_FREQUENCIES:
            raise ValueError(f"Invalid recurring frequency: {v}")
        return v.lower() if v else None
    
    @validator("tags")
    def validate_tags(cls, v):
        """Sanitize tags to prevent injection."""
        if v:
            # Remove any potentially dangerous characters
            v = re.sub(r'[<>"\';(){}]', '', v)
            # Limit number of tags
            tags = [t.strip() for t in v.split(',')][:10]
            return ','.join(tags)
        return v
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Remove potentially dangerous characters from strings."""
        # Remove HTML/script tags
        value = re.sub(r'<[^>]*>', '', value)
        # Remove SQL injection patterns
        value = re.sub(r'[;\'"\\]', '', value)
        # Remove null bytes
        value = value.replace('\x00', '')
        return value.strip()
    
    class Config:
        str_strip_whitespace = True


class TransactionValidator:
    """Validator for transaction data from spreadsheet."""
    
    def validate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and transform a transaction record.
        
        Args:
            record: Raw record from spreadsheet
            
        Returns:
            Validated and transformed record
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Parse date
            date_value = record.get("date")
            if isinstance(date_value, str):
                try:
                    date_value = datetime.strptime(date_value, "%Y-%m-%d")
                except ValueError:
                    raise ValidationError("date", "Invalid date format, expected YYYY-MM-DD")
            
            # Parse amount
            amount_value = record.get("amount")
            try:
                if isinstance(amount_value, str):
                    # Remove currency symbols and commas
                    amount_value = re.sub(r'[,$€£]', '', amount_value)
                amount_value = Decimal(str(amount_value)).quantize(Decimal("0.01"))
            except (InvalidOperation, ValueError):
                raise ValidationError("amount", "Invalid amount format")
            
            # Parse boolean
            recurring = record.get("recurring", False)
            if isinstance(recurring, str):
                recurring = recurring.upper() == "TRUE"
            
            # Build validated record
            validated_data = {
                "transaction_id": str(record.get("transaction_id", "")),
                "date": date_value,
                "category": str(record.get("category", "")),
                "subcategory": record.get("subcategory") or None,
                "description": str(record.get("description", "")),
                "amount": amount_value,
                "type": str(record.get("type", "")).lower(),
                "account": str(record.get("account", "")),
                "payment_method": record.get("payment_method") or None,
                "tags": record.get("tags") or None,
                "recurring": recurring,
                "recurring_frequency": record.get("recurring_frequency") or None,
                "notes": record.get("notes") or None,
            }
            
            # Validate with Pydantic schema
            validated = TransactionSchema(**validated_data)
            
            return validated.dict()
            
        except PydanticValidationError as e:
            errors = e.errors()
            if errors:
                first_error = errors[0]
                raise ValidationError(
                    field=".".join(str(loc) for loc in first_error["loc"]),
                    message=first_error["msg"],
                    value=first_error.get("input")
                )
            raise ValidationError("unknown", "Validation failed")
```

### app/sheets/cache.py
```python
"""
Caching layer for Google Sheets data.
"""
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

import redis
import structlog

from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class SheetsCache:
    """
    Redis-based cache for spreadsheet data.
    
    Features:
    - Automatic serialization/deserialization
    - Configurable TTL
    - Graceful degradation if Redis unavailable
    """
    
    def __init__(self):
        self.ttl = settings.cache_ttl_seconds
        self._client: Optional[redis.Redis] = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection."""
        try:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self._client.ping()
            logger.info("redis_connected", url=settings.redis_url)
        except redis.ConnectionError as e:
            logger.warning("redis_connection_failed", error=str(e))
            self._client = None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self._client:
            return None
        
        try:
            value = self._client.get(f"sheets:{key}")
            if value:
                return json.loads(value)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning("cache_get_error", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            True if successful
        """
        if not self._client:
            return False
        
        try:
            serialized = json.dumps(value, cls=DecimalEncoder)
            self._client.setex(
                f"sheets:{key}",
                ttl or self.ttl,
                serialized
            )
            return True
        except (redis.RedisError, TypeError) as e:
            logger.warning("cache_set_error", key=key, error=str(e))
            return False
    
    def invalidate(self, pattern: str = "*") -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Glob pattern for keys to invalidate
            
        Returns:
            Number of keys deleted
        """
        if not self._client:
            return 0
        
        try:
            keys = self._client.keys(f"sheets:{pattern}")
            if keys:
                return self._client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.warning("cache_invalidate_error", pattern=pattern, error=str(e))
            return 0
```

### app/main.py
```python
"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.config import get_settings
from app.auth.routes import router as auth_router
from app.dashboard.routes import router as dashboard_router
from app.database.connection import engine, Base

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("application_starting", env=settings.app_env)
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["yourdomain.com", "*.yourdomain.com"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"}
    )


# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/api/sheets/health")
async def sheets_health():
    """Check Google Sheets connectivity."""
    from app.sheets.client import GoogleSheetsClient
    
    client = GoogleSheetsClient()
    return client.health_check()
```

---

## 5. AWS Deployment

### deployment/Dockerfile
```dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ app/
COPY alembic.ini .
COPY alembic/ alembic/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

### deployment/terraform/main.tf
```hcl
# AWS Provider Configuration
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket         = "your-terraform-state-bucket"
    key            = "financial-dashboard/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "financial-dashboard"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_dns_hostnames   = true
  enable_dns_support     = true
}

# Secrets Manager for sensitive configuration
resource "aws_secretsmanager_secret" "app_secrets" {
  name = "${var.project_name}-${var.environment}-secrets"
  
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  
  secret_string = jsonencode({
    jwt_secret_key              = var.jwt_secret_key
    google_service_account_key  = var.google_service_account_key
    database_password           = random_password.db_password.result
  })
}

# Random password for RDS
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# RDS PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "6.0.0"

  identifier = "${var.project_name}-${var.environment}"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.environment == "production" ? "db.t3.medium" : "db.t3.micro"

  allocated_storage     = 20
  max_allocated_storage = 100

  db_name  = "financial_dashboard"
  username = "app_user"
  password = random_password.db_password.result
  port     = 5432

  vpc_security_group_ids = [aws_security_group.rds.id]
  subnet_ids             = module.vpc.private_subnets

  backup_retention_period = var.environment == "production" ? 7 : 1
  skip_final_snapshot     = var.environment != "production"
  deletion_protection     = var.environment == "production"

  performance_insights_enabled = var.environment == "production"
}

# Security Group for RDS
resource "aws_security_group" "rds" {
  name_prefix = "${var.project_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-${var.environment}"
  engine               = "redis"
  node_type            = var.environment == "production" ? "cache.t3.small" : "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  security_group_ids   = [aws_security_group.redis.id]
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
}

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-${var.environment}"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
}

# ECR Repository
resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-${var.environment}"

  setting {
    name  = "containerInsights"
    value = var.environment == "production" ? "enabled" : "disabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 100
    capacity_provider = var.environment == "production" ? "FARGATE" : "FARGATE_SPOT"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = var.project_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.environment == "production" ? "512" : "256"
  memory                   = var.environment == "production" ? "1024" : "512"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "app"
      image = "${aws_ecr_repository.app.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        { name = "APP_ENV", value = var.environment },
        { name = "AWS_REGION", value = var.aws_region },
        { name = "AWS_SECRETS_MANAGER_SECRET_NAME", value = aws_secretsmanager_secret.app_secrets.name },
        { name = "REDIS_URL", value = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379" },
        { name = "GOOGLE_SHEETS_SPREADSHEET_ID", value = var.google_sheets_spreadsheet_id }
      ]
      
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:database_url::"
        },
        {
          name      = "JWT_SECRET_KEY"
          valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:jwt_secret_key::"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.project_name}-${var.environment}"
  retention_in_days = var.environment == "production" ? 30 : 7
}

# Security Group for ECS
resource "aws_security_group" "ecs" {
  name_prefix = "${var.project_name}-ecs-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets

  enable_deletion_protection = var.environment == "production"
}

resource "aws_security_group" "alb" {
  name_prefix = "${var.project_name}-alb-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb_target_group" "app" {
  name        = "${var.project_name}-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = var.project_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.environment == "production" ? 2 : 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = module.vpc.private_subnets
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  depends_on = [aws_lb_listener.https]
}

# IAM Roles
resource "aws_iam_role" "ecs_execution" {
  name = "${var.project_name}-ecs-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "secrets-access"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [aws_secretsmanager_secret.app_secrets.arn]
      }
    ]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${var.project_name}-ecs-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_secrets" {
  name = "secrets-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [aws_secretsmanager_secret.app_secrets.arn]
      }
    ]
  })
}

# Outputs
output "alb_dns_name" {
  value = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "rds_endpoint" {
  value = module.rds.db_instance_endpoint
}
```

### deployment/terraform/variables.tf
```hcl
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "development"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "financial-dashboard"
}

variable "jwt_secret_key" {
  description = "JWT secret key for authentication"
  type        = string
  sensitive   = true
}

variable "google_service_account_key" {
  description = "Google service account JSON key"
  type        = string
  sensitive   = true
}

variable "google_sheets_spreadsheet_id" {
  description = "Google Sheets spreadsheet ID"
  type        = string
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
}
```

---

## 6. Step-by-Step Deployment Instructions

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **Terraform** >= 1.0 installed
3. **Docker** installed
4. **AWS CLI** configured with credentials
5. **Google Cloud Console** access

### Step 1: Google Cloud Setup

```bash
# 1. Create a Google Cloud Project
# Go to: https://console.cloud.google.com/

# 2. Enable the Google Sheets API
# Go to: APIs & Services > Library > Search "Google Sheets API" > Enable

# 3. Create a Service Account
# Go to: IAM & Admin > Service Accounts > Create Service Account
# Name: financial-dashboard-reader
# Role: Viewer

# 4. Create and download JSON key
# Click on service account > Keys > Add Key > JSON
# Save the file securely

# 5. Share your spreadsheet with the service account email
# Open your Google Sheet > Share > Add the service account email
# (e.g., financial-dashboard-reader@project-id.iam.gserviceaccount.com)
```

### Step 2: AWS Infrastructure Setup

```bash
# 1. Create S3 bucket for Terraform state
aws s3api create-bucket \
  --bucket your-terraform-state-bucket \
  --region us-east-1

# 2. Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

# 3. Request ACM certificate (for HTTPS)
aws acm request-certificate \
  --domain-name yourdomain.com \
  --validation-method DNS \
  --region us-east-1

# 4. Generate JWT secret
openssl rand -base64 64
```

### Step 3: Configure Terraform Variables

```bash
# Create terraform.tfvars
cat > deployment/terraform/terraform.tfvars << 'EOF'
aws_region                   = "us-east-1"
environment                  = "production"
project_name                 = "financial-dashboard"
google_sheets_spreadsheet_id = "YOUR_SPREADSHEET_ID"
acm_certificate_arn          = "arn:aws:acm:us-east-1:ACCOUNT:certificate/ID"
EOF

# Set sensitive variables via environment
export TF_VAR_jwt_secret_key="your-generated-secret"
export TF_VAR_google_service_account_key='{"type":"service_account",...}'
```

### Step 4: Deploy Infrastructure

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -out=tfplan

# Apply (review changes first!)
terraform apply tfplan

# Save outputs
terraform output -json > outputs.json
```

### Step 5: Build and Push Docker Image

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t financial-dashboard -f deployment/Dockerfile .

# Tag image
docker tag financial-dashboard:latest \
  ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/financial-dashboard:latest

# Push image
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/financial-dashboard:latest

# Force new ECS deployment
aws ecs update-service \
  --cluster financial-dashboard-production \
  --service financial-dashboard \
  --force-new-deployment
```

### Step 6: Database Migration

```bash
# Connect to RDS via bastion or Session Manager
# Run Alembic migrations
alembic upgrade head
```

### Step 7: Verify Deployment

```bash
# Check ECS service status
aws ecs describe-services \
  --cluster financial-dashboard-production \
  --services financial-dashboard

# Test health endpoint
curl https://yourdomain.com/health

# Check logs
aws logs tail /ecs/financial-dashboard-production --follow
```

---

## 7. Security Best Practices Implemented

1. **Secrets Management**: All sensitive data stored in AWS Secrets Manager
2. **Network Isolation**: Private subnets for ECS and RDS, NAT gateway for outbound
3. **TLS Everywhere**: HTTPS via ALB with TLS 1.3
4. **IAM Least Privilege**: Task roles have minimal required permissions
5. **Database Security**: RDS in private subnet, encrypted at rest
6. **Container Security**: Non-root user, health checks, vulnerability scanning
7. **Input Validation**: All spreadsheet data validated before processing
8. **Rate Limiting**: Respects Google Sheets API quotas
9. **Caching**: Redis caching to minimize API calls
10. **Logging**: Structured logging with CloudWatch

---

## 8. Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `APP_ENV` | Environment (development/staging/production) | No | development |
| `DATABASE_URL` | PostgreSQL connection string | Yes | - |
| `REDIS_URL` | Redis connection string | No | redis://localhost:6379 |
| `JWT_SECRET_KEY` | Secret key for JWT signing | Yes | - |
| `GOOGLE_SHEETS_SPREADSHEET_ID` | Google Sheets spreadsheet ID | Yes | - |
| `GOOGLE_SERVICE_ACCOUNT_KEY` | JSON service account key | Yes | - |
| `AWS_SECRETS_MANAGER_SECRET_NAME` | AWS secret name | No | - |
| `SHEETS_REQUESTS_PER_MINUTE` | Rate limit for Sheets API | No | 60 |
| `CACHE_TTL_SECONDS` | Cache TTL | No | 300 |

---

## 9. Troubleshooting

### Common Issues

**Spreadsheet Not Found**
- Verify spreadsheet ID is correct
- Confirm service account has access (share spreadsheet with service account email)

**Authentication Errors**
- Check service account JSON is valid
- Verify Sheets API is enabled in Google Cloud

**Rate Limiting**
- Increase cache TTL
- Reduce `SHEETS_REQUESTS_PER_MINUTE`

**Database Connection**
- Verify security groups allow ECS to RDS connection
- Check DATABASE_URL format

---

This completes the comprehensive Python-based financial dashboard implementation guide.
