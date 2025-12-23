# Financial Dashboard - API Routes Documentation

Complete API reference for the Python FastAPI backend.

---

## Authentication Endpoints

### POST /api/auth/register

Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecureP@ssw0rd!",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Validation Rules:**
- `email`: Valid email format, max 255 characters
- `password`: Min 8 characters, must include uppercase, lowercase, number, special char
- `first_name`: Max 100 characters
- `last_name`: Max 100 characters

**Response (201 Created):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Errors:**
- `400`: Invalid input data
- `409`: Email already registered

---

### POST /api/auth/login

Authenticate and receive tokens.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "SecureP@ssw0rd!"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Errors:**
- `401`: Invalid credentials
- `403`: Account disabled

---

### POST /api/auth/refresh

Refresh access token using refresh token.

**Request Body:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

---

### POST /api/auth/logout

Invalidate current session.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "message": "Successfully logged out"
}
```

---

### GET /api/auth/me

Get current user profile.

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response (200 OK):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "is_active": true,
  "created_at": "2024-01-15T10:30:00Z",
  "last_login": "2024-01-20T08:00:00Z"
}
```

---

## Dashboard Endpoints

All dashboard endpoints require authentication.

**Headers Required:**
```
Authorization: Bearer <access_token>
```

---

### GET /api/dashboard/overview

Get financial overview for current period.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `year` | int | current | Year to query |
| `month` | int | current | Month to query |

**Response (200 OK):**
```json
{
  "period": {
    "year": 2024,
    "month": 1,
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  },
  "summary": {
    "total_income": 5500.00,
    "total_expenses": 3250.75,
    "net_savings": 2249.25,
    "savings_rate": 40.89
  },
  "income_breakdown": [
    { "category": "Salary", "amount": 5000.00, "percentage": 90.91 },
    { "category": "Freelance", "amount": 500.00, "percentage": 9.09 }
  ],
  "expense_breakdown": [
    { "category": "Housing", "amount": 1800.00, "percentage": 55.38 },
    { "category": "Food", "amount": 650.00, "percentage": 20.00 },
    { "category": "Transportation", "amount": 350.00, "percentage": 10.77 }
  ],
  "budget_status": [
    {
      "category": "Food",
      "budget": 600.00,
      "spent": 650.00,
      "remaining": -50.00,
      "percentage_used": 108.33,
      "status": "over_budget"
    },
    {
      "category": "Entertainment",
      "budget": 150.00,
      "spent": 89.99,
      "remaining": 60.01,
      "percentage_used": 60.00,
      "status": "on_track"
    }
  ],
  "trend": {
    "vs_last_month": {
      "income_change": 0.00,
      "expense_change": -5.25,
      "savings_change": 125.00
    }
  }
}
```

---

### GET /api/dashboard/transactions

Get paginated transactions list.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `per_page` | int | 50 | Items per page (max 100) |
| `start_date` | date | - | Filter from date |
| `end_date` | date | - | Filter to date |
| `type` | string | - | Filter: "income" or "expense" |
| `category` | string | - | Filter by category |
| `account` | string | - | Filter by account |
| `min_amount` | float | - | Minimum amount |
| `max_amount` | float | - | Maximum amount |
| `search` | string | - | Search in description |
| `sort_by` | string | date | Sort field |
| `sort_order` | string | desc | asc or desc |

**Response (200 OK):**
```json
{
  "data": [
    {
      "transaction_id": "TXN-2024-001",
      "date": "2024-01-15",
      "category": "Food",
      "subcategory": "Groceries",
      "description": "Weekly grocery shopping",
      "amount": 125.50,
      "type": "expense",
      "account": "Checking",
      "payment_method": "debit",
      "tags": ["essential", "food"],
      "recurring": false,
      "notes": null
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total_items": 156,
    "total_pages": 4,
    "has_next": true,
    "has_prev": false
  },
  "filters_applied": {
    "type": null,
    "category": null,
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-31"
    }
  }
}
```

---

### GET /api/dashboard/accounts

Get all accounts with current balances.

**Response (200 OK):**
```json
{
  "accounts": [
    {
      "account_id": "ACC-001",
      "account_name": "Primary Checking",
      "account_type": "checking",
      "institution": "Chase Bank",
      "current_balance": 5432.10,
      "currency": "USD"
    },
    {
      "account_id": "ACC-002",
      "account_name": "Emergency Savings",
      "account_type": "savings",
      "institution": "Ally Bank",
      "current_balance": 15000.00,
      "currency": "USD"
    }
  ],
  "totals": {
    "net_worth": 44831.85,
    "by_type": {
      "checking": 5432.10,
      "savings": 15000.00,
      "credit": -850.25,
      "investment": 25000.00,
      "cash": 250.00
    }
  }
}
```

---

### GET /api/dashboard/budgets

Get budget information for specified period.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `year` | int | current | Year |
| `month` | int | current | Month |

**Response (200 OK):**
```json
{
  "period": {
    "year": 2024,
    "month": 1
  },
  "budgets": [
    {
      "budget_id": "BUD-2024-01-FOOD",
      "category": "Food",
      "monthly_limit": 600.00,
      "spent": 485.25,
      "remaining": 114.75,
      "percentage_used": 80.88,
      "status": "on_track",
      "projected_end_of_month": 620.50,
      "daily_average": 31.05
    }
  ],
  "summary": {
    "total_budget": 4050.00,
    "total_spent": 2890.50,
    "total_remaining": 1159.50,
    "overall_percentage": 71.37,
    "categories_over_budget": 1,
    "categories_on_track": 6,
    "categories_under_budget": 1
  }
}
```

---

### GET /api/dashboard/trends

Get financial trends over time.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | 12_months | 3_months, 6_months, 12_months, ytd |
| `granularity` | string | monthly | daily, weekly, monthly |

**Response (200 OK):**
```json
{
  "period": "12_months",
  "granularity": "monthly",
  "data_points": [
    {
      "period": "2023-02",
      "income": 5500.00,
      "expenses": 3100.00,
      "savings": 2400.00,
      "savings_rate": 43.64
    },
    {
      "period": "2023-03",
      "income": 5500.00,
      "expenses": 3450.00,
      "savings": 2050.00,
      "savings_rate": 37.27
    }
  ],
  "averages": {
    "monthly_income": 5500.00,
    "monthly_expenses": 3225.50,
    "monthly_savings": 2274.50,
    "savings_rate": 41.35
  },
  "trends": {
    "income_trend": "stable",
    "expense_trend": "decreasing",
    "savings_trend": "increasing"
  }
}
```

---

### GET /api/dashboard/categories

Get spending breakdown by category.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `year` | int | current | Year |
| `month` | int | - | Month (optional, for full year if not specified) |
| `type` | string | expense | income or expense |

**Response (200 OK):**
```json
{
  "type": "expense",
  "period": {
    "year": 2024,
    "month": 1
  },
  "categories": [
    {
      "category": "Housing",
      "total": 1800.00,
      "percentage": 55.38,
      "transaction_count": 2,
      "average_transaction": 900.00,
      "subcategories": [
        { "name": "Rent", "total": 1800.00 }
      ]
    },
    {
      "category": "Food",
      "total": 650.00,
      "percentage": 20.00,
      "transaction_count": 12,
      "average_transaction": 54.17,
      "subcategories": [
        { "name": "Groceries", "total": 450.00 },
        { "name": "Dining Out", "total": 150.00 },
        { "name": "Delivery", "total": 50.00 }
      ]
    }
  ],
  "total": 3250.75
}
```

---

### POST /api/dashboard/refresh

Force refresh data from Google Sheets (bypasses cache).

**Response (200 OK):**
```json
{
  "message": "Data refreshed successfully",
  "transactions_count": 156,
  "accounts_count": 5,
  "budgets_count": 8,
  "refreshed_at": "2024-01-20T10:30:00Z"
}
```

**Rate Limited:** Max 1 request per 5 minutes per user.

---

## Health & Status Endpoints

### GET /health

Basic health check (no auth required).

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

---

### GET /api/sheets/health

Check Google Sheets connectivity.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "spreadsheet_title": "Personal Finance Tracker",
  "last_updated": "2024-01-20T08:00:00Z",
  "worksheets": [
    "Financial_Tracker",
    "Accounts",
    "Budgets"
  ]
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "error": "Spreadsheet not found or access denied"
}
```

---

## Error Response Format

All error responses follow this format:

```json
{
  "detail": "Human-readable error message",
  "error_code": "ERROR_CODE",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_CREDENTIALS` | 401 | Wrong email or password |
| `TOKEN_EXPIRED` | 401 | Access token expired |
| `TOKEN_INVALID` | 401 | Invalid token format |
| `USER_DISABLED` | 403 | Account is disabled |
| `PERMISSION_DENIED` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Input validation failed |
| `RATE_LIMITED` | 429 | Too many requests |
| `SHEETS_UNAVAILABLE` | 503 | Cannot connect to Google Sheets |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/auth/login` | 10 requests per minute |
| `/api/auth/register` | 5 requests per minute |
| `/api/dashboard/*` | 60 requests per minute |
| `/api/dashboard/refresh` | 1 request per 5 minutes |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705750800
```
