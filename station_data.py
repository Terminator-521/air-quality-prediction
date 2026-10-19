"""Optional connector for an approved official air-quality monitor feed.

Set CPCB_AQI_URL only when you have permission to use an endpoint. It must
return a JSON object with station_name, aqi, observed_at and optional
distance_km/pollutants fields. Keeping this connector opt-in prevents the app
from misrepresenting a scraped or unlicensed feed as an official reading.
"""

import requests

from config import OFFICIAL_MONITOR_URL


class OfficialMonitorError(Exception):
    pass


def configured():
    return bool(OFFICIAL_MONITOR_URL)


def fetch_nearest_monitor(lat, lon):
    """Return a normalized official-monitor reading, or None when unconfigured."""
    if not configured():
        return None
    try:
        response = requests.get(OFFICIAL_MONITOR_URL, params={"latitude": lat, "longitude": lon}, timeout=10)
        response.raise_for_status()
        reading = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise OfficialMonitorError("Official monitor data could not be retrieved.") from exc
    required = {"station_name", "aqi", "observed_at"}
    if not isinstance(reading, dict) or not required.issubset(reading):
        raise OfficialMonitorError("Official monitor endpoint returned an unsupported response.")
    return reading
