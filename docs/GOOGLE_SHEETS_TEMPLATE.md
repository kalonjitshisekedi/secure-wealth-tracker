# Financial Dashboard - Google Sheets Template

This document provides the exact structure for setting up your Google Sheets financial tracker.

---

## Quick Setup

1. Create a new Google Sheets document
2. Create the following worksheets (tabs):
   - `Financial_Tracker` (main transactions)
   - `Accounts` (bank accounts)
   - `Budgets` (monthly budgets)

---

## Worksheet 1: Financial_Tracker

### Header Row (Row 1)

Copy and paste this into Row 1, cells A1 through O1:

```
transaction_id | date | category | subcategory | description | amount | type | account | payment_method | tags | recurring | recurring_frequency | notes | created_at | updated_at
```

### Column Specifications

| Column | Header | Data Type | Example | Validation |
|--------|--------|-----------|---------|------------|
| A | transaction_id | Text | `TXN-2024-001` | Unique ID |
| B | date | Date | `2024-01-15` | YYYY-MM-DD format |
| C | category | Text | `Food` | From category list |
| D | subcategory | Text | `Groceries` | Optional |
| E | description | Text | `Weekly grocery shopping` | Max 500 chars |
| F | amount | Currency | `$125.50` | Positive number |
| G | type | Text | `expense` | "income" or "expense" |
| H | account | Text | `Checking` | From accounts list |
| I | payment_method | Text | `debit` | cash/credit/debit/transfer |
| J | tags | Text | `essential,food` | Comma-separated |
| K | recurring | Boolean | `FALSE` | TRUE or FALSE |
| L | recurring_frequency | Text | `monthly` | weekly/monthly/yearly |
| M | notes | Text | `Bought for party` | Max 1000 chars |
| N | created_at | DateTime | `2024-01-15T10:30:00` | Auto-generated |
| O | updated_at | DateTime | `2024-01-15T10:30:00` | Auto-updated |

### Sample Data (Rows 2-6)

```csv
TXN-2024-001,2024-01-15,Food,Groceries,Weekly grocery shopping,125.50,expense,Checking,debit,"essential,food",FALSE,,Fresh produce and staples,2024-01-15T10:30:00,2024-01-15T10:30:00
TXN-2024-002,2024-01-14,Transportation,Fuel,Gas station fill-up,45.00,expense,Credit Card,credit,"car,essential",FALSE,,,2024-01-14T08:15:00,2024-01-14T08:15:00
TXN-2024-003,2024-01-01,Salary,,Monthly salary deposit,5500.00,income,Checking,transfer,"salary,primary",TRUE,monthly,January 2024 salary,2024-01-01T09:00:00,2024-01-01T09:00:00
TXN-2024-004,2024-01-10,Housing,Rent,Monthly rent payment,1800.00,expense,Checking,transfer,"rent,essential",TRUE,monthly,January rent,2024-01-10T00:00:00,2024-01-10T00:00:00
TXN-2024-005,2024-01-12,Entertainment,Streaming,Netflix subscription,15.99,expense,Credit Card,credit,"subscription,entertainment",TRUE,monthly,,2024-01-12T12:00:00,2024-01-12T12:00:00
```

### Data Validation Rules (Google Sheets)

Apply these data validation rules in Google Sheets:

**Column C (category)** - Dropdown list:
```
Salary,Freelance,Investments,Rental Income,Dividends,Refunds,Gifts,Other Income,Housing,Transportation,Food,Healthcare,Entertainment,Shopping,Personal,Financial,Other Expenses
```

**Column G (type)** - Dropdown list:
```
income,expense
```

**Column I (payment_method)** - Dropdown list:
```
cash,credit,debit,transfer,check
```

**Column K (recurring)** - Dropdown list:
```
TRUE,FALSE
```

**Column L (recurring_frequency)** - Dropdown list:
```
weekly,monthly,yearly
```

---

## Worksheet 2: Accounts

### Header Row (Row 1)

```
account_id | account_name | account_type | institution | current_balance | currency
```

### Column Specifications

| Column | Header | Data Type | Example |
|--------|--------|-----------|---------|
| A | account_id | Text | `ACC-001` |
| B | account_name | Text | `Primary Checking` |
| C | account_type | Text | `checking` |
| D | institution | Text | `Chase Bank` |
| E | current_balance | Currency | `5432.10` |
| F | currency | Text | `USD` |

### Sample Data

```csv
ACC-001,Primary Checking,checking,Chase Bank,5432.10,USD
ACC-002,Emergency Savings,savings,Ally Bank,15000.00,USD
ACC-003,Rewards Credit Card,credit,Capital One,-850.25,USD
ACC-004,Investment Account,investment,Fidelity,25000.00,USD
ACC-005,Cash Wallet,cash,,250.00,USD
```

### Data Validation Rules

**Column C (account_type)** - Dropdown list:
```
checking,savings,credit,investment,cash
```

**Column F (currency)** - Dropdown list:
```
USD,EUR,GBP,CAD,AUD,JPY
```

---

## Worksheet 3: Budgets

### Header Row (Row 1)

```
budget_id | category | monthly_limit | year | month
```

### Column Specifications

| Column | Header | Data Type | Example |
|--------|--------|-----------|---------|
| A | budget_id | Text | `BUD-2024-01-FOOD` |
| B | category | Text | `Food` |
| C | monthly_limit | Currency | `600.00` |
| D | year | Number | `2024` |
| E | month | Number | `1` |

### Sample Data (January 2024 budgets)

```csv
BUD-2024-01-HOUSING,Housing,2000.00,2024,1
BUD-2024-01-FOOD,Food,600.00,2024,1
BUD-2024-01-TRANSPORT,Transportation,300.00,2024,1
BUD-2024-01-HEALTH,Healthcare,200.00,2024,1
BUD-2024-01-ENTERTAIN,Entertainment,150.00,2024,1
BUD-2024-01-SHOPPING,Shopping,200.00,2024,1
BUD-2024-01-PERSONAL,Personal,100.00,2024,1
BUD-2024-01-FINANCIAL,Financial,500.00,2024,1
```

---

## Sheet Formatting Recommendations

### Conditional Formatting

Apply these conditional formatting rules for better visualization:

1. **Expense rows (red highlight)**
   - Range: Column G
   - Condition: Text equals "expense"
   - Format: Light red background

2. **Income rows (green highlight)**
   - Range: Column G
   - Condition: Text equals "income"
   - Format: Light green background

3. **Recurring transactions (bold)**
   - Range: Column K
   - Condition: Text equals "TRUE"
   - Format: Bold text

4. **Negative balances (red text)**
   - Range: Column E (Accounts sheet)
   - Condition: Less than 0
   - Format: Red text

### Number Formatting

- **Amount columns**: Currency format with 2 decimal places
- **Date columns**: Date format (YYYY-MM-DD)
- **Percentage columns**: Percentage format

---

## Formulas for Summary Sheet (Optional)

Create a `Summary` worksheet with these formulas:

### Total Income (Current Month)
```excel
=SUMIFS(Financial_Tracker!F:F, Financial_Tracker!G:G, "income", Financial_Tracker!B:B, ">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1), Financial_Tracker!B:B, "<"&DATE(YEAR(TODAY()),MONTH(TODAY())+1,1))
```

### Total Expenses (Current Month)
```excel
=SUMIFS(Financial_Tracker!F:F, Financial_Tracker!G:G, "expense", Financial_Tracker!B:B, ">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1), Financial_Tracker!B:B, "<"&DATE(YEAR(TODAY()),MONTH(TODAY())+1,1))
```

### Net Savings (Current Month)
```excel
=TotalIncome-TotalExpenses
```

### Spending by Category (Current Month)
```excel
=SUMIFS(Financial_Tracker!F:F, Financial_Tracker!G:G, "expense", Financial_Tracker!C:C, "Food", Financial_Tracker!B:B, ">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1))
```

### Budget Remaining
```excel
=Budgets!C2 - SUMIFS(Financial_Tracker!F:F, Financial_Tracker!C:C, Budgets!B2, Financial_Tracker!B:B, ">="&DATE(Budgets!D2,Budgets!E2,1))
```

---

## Service Account Access Setup

1. **Get Service Account Email**
   - Go to Google Cloud Console > IAM & Admin > Service Accounts
   - Find your service account email (e.g., `financial-reader@project-id.iam.gserviceaccount.com`)

2. **Share Spreadsheet**
   - Open your Google Sheet
   - Click "Share" button
   - Enter the service account email
   - Set permission to "Viewer" (read-only recommended)
   - Uncheck "Notify people" (service accounts don't receive emails)
   - Click "Share"

3. **Get Spreadsheet ID**
   - Open your spreadsheet
   - Look at the URL: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`
   - Copy the `SPREADSHEET_ID` portion

---

## Data Entry Best Practices

1. **Use consistent categories** - Stick to the predefined category list
2. **Always include transaction ID** - Use a consistent format like `TXN-YYYY-NNN`
3. **Use ISO date format** - YYYY-MM-DD for consistent parsing
4. **Keep descriptions brief** - Max 500 characters, be descriptive
5. **Tag appropriately** - Use comma-separated tags for filtering
6. **Mark recurring correctly** - Helps with budget forecasting
7. **Update balances regularly** - Keep Accounts sheet current

---

## Troubleshooting

### API Cannot Read Data
- Verify service account has access to the spreadsheet
- Check spreadsheet ID is correct
- Ensure worksheets are named exactly as specified

### Data Validation Errors
- Check date format is YYYY-MM-DD
- Verify amounts are positive numbers
- Ensure type is exactly "income" or "expense" (lowercase)

### Missing Columns
- The API expects all columns in order
- Don't skip or reorder columns
- Empty cells are okay, missing columns are not
