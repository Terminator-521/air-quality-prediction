"""Email delivery for scheduled AQI threshold notifications."""

import smtplib
from email.message import EmailMessage

from config import AQI_ALERT_FROM, AQI_SMTP_HOST, AQI_SMTP_PASSWORD, AQI_SMTP_PORT, AQI_SMTP_USER


def email_alert(recipient, location_name, threshold, peak_aqi, forecast_through):
    """Send a single plain-text alert through configured STARTTLS SMTP."""
    if not AQI_SMTP_HOST or not AQI_ALERT_FROM:
        raise RuntimeError("AQI_SMTP_HOST and AQI_ALERT_FROM must be configured to send alerts.")
    message = EmailMessage()
    message["Subject"] = f"Air-quality alert for {location_name}: forecast AQI {peak_aqi:.0f}"
    message["From"] = AQI_ALERT_FROM
    message["To"] = recipient
    message.set_content(
        f"The current forecast for {location_name} reaches AQI {peak_aqi:.0f}, "
        f"above your threshold of {threshold}.\n\n"
        f"Forecast coverage: through {forecast_through}.\n"
        "AQI values are model estimates, not official monitor readings. Check local "
        "public-health guidance and nearby regulatory monitors for decisions.\n"
    )
    with smtplib.SMTP(AQI_SMTP_HOST, AQI_SMTP_PORT, timeout=20) as smtp:
        smtp.starttls()
        if AQI_SMTP_USER:
            smtp.login(AQI_SMTP_USER, AQI_SMTP_PASSWORD)
        smtp.send_message(message)
