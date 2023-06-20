from pytz import timezone
from datetime import datetime


def get_call_metadata() -> dict:
    """Get the timestamp of when the API is called"""

    # Create timezone (fixed)
    tz = timezone("Europe/Dublin")

    # Get timestamp in the form of str
    called_at = datetime.now(tz).isoformat()

    # Add specifics
    called_at = called_at + " (IE time)"

    return {"called": called_at}
