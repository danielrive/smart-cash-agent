import requests
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Set global precision for Decimal (more than enough for FX rates)
getcontext().prec = 12

def get_exchange_rate(base_currency: str, target_currency: str) -> Optional[float]:
    """
    Fetch the current exchange rate between two currencies using ExchangeRate API.

    Thanks to https://www.exchangerate-api.com Rates By Exchange Rate API 

    Args:
        base_currency (str): The currency code to convert from (e.g., "EUR").
        target_currency (str): The currency code to convert to (e.g., "USD").

    Returns:
        float: The exchange rate (1 base_currency = X target_currency).
        None: If the request fails or the API does not return success.
    """
    base_url = "https://open.er-api.com/v6/latest/"
    try:
        logger.info(f"Fetching exchange rate for {base_currency} to {target_currency}")
        logger.info(f"calling {base_url}{base_currency}")
        response = requests.get(f"{base_url}{base_currency}", timeout=10)
        logger.info(f"response {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if data.get("result") != "success":
            logger.error(f"API did not return success: {data}")
            return None

        rate = data["rates"].get(target_currency)
        if rate is None:
            logger.error(f"Rate not found for {target_currency}")
            return None

        return Decimal(str(rate))

    except requests.RequestException as e:
        logger.error(f"Error fetching exchange rate: {e}")
        return None


def convert_currency(value: float, base_currency: str, target_currency: str,rounding: str = ROUND_HALF_UP) -> Optional[float]:
    """
    Convert an amount from one currency to another using live exchange rates provided by By Exchange Rate API, check https://www.exchangerate-api.com

    Args:
        value (float): Amount in the base currency.
        base_currency (str): The currency code to convert from.
        target_currency (str): The currency code to convert to.
        rounding (str): Decimal rounding strategy (default: ROUND_HALF_UP).

    Returns:
        float: Converted amount.
        None: If conversion fails.
    """
    rate = get_exchange_rate(base_currency, target_currency)
    if rate is None:
        logger.error(f"Failed to get exchange rate for {base_currency} to {target_currency}")
        return None
    
    amount = Decimal(str(value)) * rate
    # Round to 2 decimal places (standard for currencies)
    return amount.quantize(Decimal("0.01"), rounding=rounding)
