"""Email alert agent using SMTP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import imghdr
import smtplib
from typing import List


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_pass: str
    alert_from: str
    alert_to: List[str]


class EmailAlertAgent:
    def __init__(self, config: EmailConfig | None = None) -> None:
        self.config = config or self._load_from_env()

    @staticmethod
    def _load_from_env() -> EmailConfig:
        host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASS", "")
        alert_from = os.getenv("ALERT_FROM", user)
        alert_to_raw = os.getenv("ALERT_TO", "")
        alert_to = [addr.strip() for addr in alert_to_raw.split(",") if addr.strip()]

        if not user or not password or not alert_from or not alert_to:
            raise ValueError("Email environment variables are not fully configured")

        return EmailConfig(
            smtp_host=host,
            smtp_port=port,
            smtp_user=user,
            smtp_pass=password,
            alert_from=alert_from,
            alert_to=alert_to,
        )

    def send_alert(self, image_path: str, score: float, matched_name: str | None) -> None:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = "Wanted Person Detected"
        if matched_name:
            subject = f"Wanted Person Detected: {matched_name}"

        message = MIMEMultipart()
        message["From"] = self.config.alert_from
        message["To"] = ", ".join(self.config.alert_to)
        message["Subject"] = subject

        body = (
            f"Time: {now}\n"
            f"Match Score: {score:.3f}\n"
            f"Matched Name: {matched_name or 'Unknown'}\n"
        )
        message.attach(MIMEText(body, "plain"))

        with open(image_path, "rb") as file_handle:
            image_data = file_handle.read()
        subtype = imghdr.what(None, h=image_data) or "octet-stream"
        attachment = MIMEApplication(image_data, _subtype=subtype)
        attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
        message.attach(attachment)

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            server.starttls()
            server.login(self.config.smtp_user, self.config.smtp_pass)
            server.sendmail(self.config.alert_from, self.config.alert_to, message.as_string())
