"""Central settings for the air-quality dashboard.

Values that may differ between deployments are read from environment variables.
Keep credentials in environment variables or Streamlit secrets, never here.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

GEOCODE_URL = os.getenv("AQI_GEOCODE_URL", "https://geocoding-api.open-meteo.com/v1/search")
AIR_QUALITY_URL = os.getenv("AQI_AIR_QUALITY_URL", "https://air-quality-api.open-meteo.com/v1/air-quality")
LIVE_CACHE_SECONDS = int(os.getenv("AQI_LIVE_CACHE_SECONDS", "900"))
PAST_DAYS = int(os.getenv("AQI_PAST_DAYS", "2"))
FORECAST_DAYS = int(os.getenv("AQI_FORECAST_DAYS", "4"))
WEATHER_URL = os.getenv("AQI_WEATHER_URL", "https://api.open-meteo.com/v1/forecast")
# Supply a licensed/approved endpoint returning the normalized monitor schema in
# station_data.py. CPCB does not provide a public unauthenticated endpoint used
# by this project.
OFFICIAL_MONITOR_URL = os.getenv("CPCB_AQI_URL", "").strip()
_database_path = Path(os.getenv("AQI_DATABASE_PATH", str(DATA_DIR / "air_quality.sqlite")))
AIR_QUALITY_DB_PATH = _database_path if _database_path.is_absolute() else PROJECT_ROOT / _database_path
FORECAST_TRACKER_PATH = AIR_QUALITY_DB_PATH
