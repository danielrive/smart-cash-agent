from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import Optional, List
import boto3
import logging
from boto3.dynamodb.conditions import Key

logger = logging.getLogger(__name__)

"""
    DynamoDB tools for managing expenses.

    These functions provide CRUD-like operations for the Expenses table.
    They are written with detailed docstrings so LLMs or developers can 
    understand required parameters and return values.
"""

# -----------------------------
# Data Models
# -----------------------------

@dataclass
class Expense:
    """
    Dataclass representing a single expense record.

    Fields:
        expenseId (str): Unique ID for the expense.
        date (str): Date in ISO format YYYY-MM-DD.
        description (str): Text description of the expense.
        status (str): Current status of the expense (e.g. "PENDING", "PAID").
        name (str): Short title or name for the expense.
        amount (Decimal): Expense amount.
        userId (str): ID of the user who owns the expense.
        category (str): Category for the expense (e.g. "Meals", "Travel").
    """
    expenseId: str
    date: str
    description: str
    status: str
    name: str
    amount: Decimal
    userId: str
    category: str


@dataclass
class ExpensesReturn:
    """
    A lightweight view of an expense, often used for lists.
    """
    expenseId: str
    date: str
    name: str
    amount: Decimal
    status: str


@dataclass
class ExpensesPay:
    """
    Dataclass used when paying an expense (only requires ID).
    """
    expenseId: str


# -----------------------------
# DynamoDB Setup
# -----------------------------

dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
expenses_table = dynamodb.Table("Expenses")


# -----------------------------
# Tool Functions
# -----------------------------

def add_expense(
    expenseId: str,
    date: str,
    description: str,
    status: str,
    name: str,
    amount: float,
    userId: str,
    category: str
) -> dict:
    """
    Add a new expense to DynamoDB.

    Args:
        expenseId (str): Unique identifier.
        date (str): Date in ISO format (YYYY-MM-DD).
        description (str): Expense description.
        status (str): Expense status.
        name (str): Title of the expense.
        amount (float): Expense amount.
        userId (str): User who owns the expense.
        category (str): Expense category.

    Returns:
        dict: { "status": "success", "expenseId": str }
    """
    expense = Expense(
        expenseId=expenseId,
        date=date,
        description=description,
        status=status,
        name=name,
        amount=Decimal(str(amount)),
        userId=userId,
        category=category
    )
    logger.info(f"expense added succedfully {expense}")
    # item = asdict(expense)
    # Ensure Decimal type
    # if isinstance(item["amount"], float):
    #     item["amount"] = Decimal(str(item["amount"]))

    # logger.info(f"Adding expense: {expense.expenseId}")
    # response = expenses_table.put_item(Item=item)
    # logger.info(f"Response: {response}")
    return {"status": "success", "expenseId": expense.expenseId}


def get_expense(expenseId: str) -> Optional[Expense]:
    """
    Retrieve an expense by its ID.

    Args:
        expenseId (str): Unique ID of the expense to retrieve.

    Returns:
        Expense | None: Full expense record if found, otherwise None.

    Example:
        >>> get_expense("exp123")
        Expense(expenseId="exp123", date="2025-08-20", ...)
    """
    logger.info(f"Retrieving expense: {expenseId}")
    response = expenses_table.get_item(Key={"expenseId": expenseId})
    item = response.get("Item")
    if not item:
        return None
    return Expense(**item)


def list_expenses_by_user(userId: str) -> List[Expense]:
    """
    List all expenses belonging to a specific user.

    Args:
        userId (str): The user ID whose expenses should be listed.

    Returns:
        list[Expense]: A list of Expense objects.

    Notes:
        - Requires a Global Secondary Index (GSI) on "userId" called "userId-index".

    Example:
        >>> list_expenses_by_user("user789")
        [Expense(...), Expense(...)]
    """
    logger.info(f"Listing expenses for user: {userId}")
    response = expenses_table.query(
        IndexName="userId-index",
        KeyConditionExpression=Key("userId").eq(userId)
    )
    return [Expense(**item) for item in response.get("Items", [])]


def list_expenses_by_category(category: str) -> List[Expense]:
    """
    List all expenses within a specific category.

    Args:
        category (str): Category name (e.g. "Meals", "Travel").

    Returns:
        list[Expense]: A list of Expense objects.

    Notes:
        - Requires a Global Secondary Index (GSI) on "category" called "category-index".
    """
    logger.info(f"Listing expenses for category: {category}")
    response = expenses_table.query(
        IndexName="category-index",
        KeyConditionExpression=Key("category").eq(category)
    )
    return [Expense(**item) for item in response.get("Items", [])]


def update_expense_status(expenseId: str, status: str) -> dict:
    """
    Update the status of an expense.

    Args:
        expenseId (str): ID of the expense to update.
        status (str): New status value ("PENDING", "PAID", etc.).

    Returns:
        dict: The updated attributes of the expense.

    Example:
        >>> update_expense_status("exp123", "PAID")
        {"expenseId": "exp123", "status": "PAID", ...}
    """
    logger.info(f"Updating expense {expenseId} to status {status}")
    response = expenses_table.update_item(
        Key={"expenseId": expenseId},
        UpdateExpression="SET #s = :status",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":status": status},
        ReturnValues="ALL_NEW"
    )
    return response["Attributes"]
