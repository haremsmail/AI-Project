"""Email alert agent using SMTP."""

from __future__ import annotations

import os
""" used to read environment variable"""
from dataclasses import dataclass
""" used to create simple data clase"""
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
""" used to create email massege contains text and image """
import imghdr
""" is detect image type """
import smtplib
""" used to send email using smtp protocol"""
from typing import List


@dataclass
class EmailConfig:
    """ store email seeting """
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_pass: str
    alert_from: str
    """ this is a sender email """
    alert_to: List[str]
    """" this is reciver email"""


class EmailAlertAgent:
    def __init__(self, config: EmailConfig | None = None) -> None:
        self.config = config or self._load_from_env()
        """ if config is provided use it otherwise load from system environment variable using _load_from_env method which read all email setting from environment variable and then create EmailConfig object and return it and then save it in self.config variable to be used later when sending email alert"""

    @staticmethod
    def _load_from_env() -> EmailConfig:
        """ statim method load email seting"""
        host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        port = int(os.getenv("SMTP_PORT", "587"))
        user = os.getenv("SMTP_USER", "")
        password = os.getenv("SMTP_PASS", "")
        alert_from = os.getenv("ALERT_FROM", user)
        alert_to_raw = os.getenv("ALERT_TO", "")
        alert_to = [addr.strip() for addr in alert_to_raw.split(",") if addr.strip()]
        """" convert text to list
        police@gmail.com
,admin@gmail.com
"

["police@gmail.com
","admin@gmail.com
"]"""

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
    """ this is means return seeting"""

    def send_alert(self, image_path: str, score: float, matched_name: str | None) -> None:
        """ this is used to sending email alert """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = "Wanted Person Detected"
        """" this is default email tile"""
        """ meaning get current time"""
        if matched_name:
            subject = f"Wanted Person Detected: {matched_name}"

        message = MIMEMultipart()
        """ this is create email massege"""
        message["From"] = self.config.alert_from
        message["To"] = ", ".join(self.config.alert_to)
        message["Subject"] = subject

        body = (
            f"Time: {now}\n"
            f"Match Score: {score:.3f}\n"
            f"Matched Name: {matched_name or 'Unknown'}\n"
        )
        message.attach(MIMEText(body, "plain"))
        """ add text to email massege"""

        with open(image_path, "rb") as file_handle:
            """"" open image file this maeans bineary"""
            image_data = file_handle.read()
        subtype = imghdr.what(None, h=image_data) or "octet-stream"
        """ find image type"""
        attachment = MIMEApplication(image_data, _subtype=subtype)
        """ preapare image file"""
        attachment.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
        """ add header to image file to be attached in email and then attach it to email massege"""
        message.attach(attachment)
        """" added image to email massege"""

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            """ connect to smtp server and then send email alert"""
            server.starttls()
            """ secure connnection"""
            server.login(self.config.smtp_user, self.config.smtp_pass)
            server.sendmail(self.config.alert_from, self.config.alert_to, message.as_string())
            """ this is used to login and sending email"""
