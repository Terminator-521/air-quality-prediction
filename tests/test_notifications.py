import notifications


class FakeSMTP:
    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.started_tls = False
        self.message = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, message):
        self.message = message


def test_email_alert_uses_tls_and_reports_threshold(monkeypatch):
    smtp = None

    def fake_smtp(host, port, timeout):
        nonlocal smtp
        smtp = FakeSMTP(host, port, timeout)
        return smtp

    monkeypatch.setattr(notifications, "AQI_SMTP_HOST", "smtp.example.com")
    monkeypatch.setattr(notifications, "AQI_SMTP_PORT", 587)
    monkeypatch.setattr(notifications, "AQI_SMTP_USER", "sender")
    monkeypatch.setattr(notifications, "AQI_SMTP_PASSWORD", "secret")
    monkeypatch.setattr(notifications, "AQI_ALERT_FROM", "alerts@example.com")
    monkeypatch.setattr(notifications.smtplib, "SMTP", fake_smtp)

    notifications.email_alert("person@example.com", "Pune, India", 150, 203, "2026-10-22 15:00")

    assert smtp.started_tls
    assert smtp.logged_in == ("sender", "secret")
    assert smtp.message["To"] == "person@example.com"
    assert "203" in smtp.message.get_content()
