# Wanted Detection Multi-Agent System

Face detection system with multi-agent architecture for webcam-based wanted person detection.

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
```

## Setup

1. Create `.env` file:
```
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

2. Add wanted person images to `wanted_database/wanted_persons/`

3. Verify setup:
```bash
python test_diagnostic.py
```

## Run

```bash
# Start detection
python main.py --image test1.jpg
# for webcam 
python main.py --webcam

# View reports
python view_reports.py
```

## Agents

- **Face Detection** - MTCNN detector from webcam
- **Face Embedding** - Convert faces to vectors
- **Wanted Comparison** - Match against database
- **Email Alert** - Send notifications
- **Reporting** - Generate JSON reports
