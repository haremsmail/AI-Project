"""Test EmailAlertAgent."""

import os

print("[Test] EmailAlertAgent requires environment variables:")
print("  SMTP_HOST (default: smtp.gmail.com)")
print("  SMTP_PORT (default: 587)")
print("  SMTP_USER")
print("  SMTP_PASS")
print("  ALERT_FROM")
print("  ALERT_TO (comma-separated)")

print("\n[Test] Current environment:")
print(f"  SMTP_USER: {os.getenv('SMTP_USER', 'NOT SET')}")
print(f"  SMTP_PASS: {'SET' if os.getenv('SMTP_PASS') else 'NOT SET'}")
print(f"  ALERT_TO: {os.getenv('ALERT_TO', 'NOT SET')}")

if not os.getenv('SMTP_USER') or not os.getenv('SMTP_PASS'):
    print("\n[Info] Email not configured. To test:")
    print("  setx SMTP_USER your_email@gmail.com")
    print("  setx SMTP_PASS your_app_password")
    print("  setx ALERT_TO recipient@example.com")
    print("  (restart PowerShell after setx)")
else:
    print("\n[Test] Loading EmailAlertAgent...")
    from agents.email_alert_agent import EmailAlertAgent
    try:
        agent = EmailAlertAgent()
        print("[Test] EmailAlertAgent initialized successfully")
        print("[Info] Not sending actual email in test mode")
    except Exception as e:
        print(f"[Error] {e}")
