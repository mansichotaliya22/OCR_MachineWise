<<<<<<< HEAD
# Industrial OCR Monitoring System

> **Stack:** YOLOv8 · PaddleOCR · OpenCV · FastAPI · MQTT (EMQX) · MongoDB · Docker

Detects and reads **engraved / laser-etched codes** on industrial machine parts. The system uses YOLOv8 to locate the code region, OpenCV to preprocess the crop, and PaddleOCR to extract the text — all wired together with an async FastAPI backend, MQTT message bus, and a dark-theme browser dashboard.

---

## Project Structure

```
OCR_MQTT/
├── master_dataset/         # YOLOv8-format training data (bring your own)
├── detector/
│   ├── train.py            # YOLOv8 training script
│   ├── predict.py          # YOLOv8 inference wrapper
│   └── best.pt             # Trained model (generated after training)
├── preprocessing/
│   └── preprocess.py       # OpenCV preprocessing pipeline
├── inference/
│   ├── crop_roi.py         # Crop detected bounding boxes
│   └── pipeline.py         # End-to-end OCR pipeline
├── mqtt/
│   ├── publisher.py        # MQTT publisher (EMQX)
│   └── subscriber.py       # MQTT subscriber → MongoDB
├── database/
│   ├── mongo.py            # Async Motor data-access layer
│   └── models.py           # Pydantic MongoDB document models
├── backend/
│   ├── main.py             # FastAPI app + lifespan hooks
│   ├── routes.py           # HTTP endpoints
│   ├── schemas.py          # Request/response Pydantic schemas
│   └── services.py         # Business logic layer
├── frontend/
│   ├── index.html          # Dashboard layout
│   ├── style.css           # Dark-theme styles
│   └── app.js              # Upload, display, polling logic
├── images/input/           # Uploaded images (runtime)
├── results/                # Annotated detection images (runtime)
├── models/                 # YOLOv8 training run artifacts
├── config.py               # Central settings (Pydantic BaseSettings)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Quick Start (Docker)

### 1. Prerequisites

- Docker Desktop ≥ 24 (or Docker Engine + Compose v2)
- Your trained `detector/best.pt` **or** run Step 2 first

### 2. Copy environment file

```bash
cp .env.example .env
# Edit .env if needed (defaults work for local Docker Compose)
```

### 3. Train the model (first time only)

```bash
# Run outside Docker so GPU/CPU is available directly
pip install ultralytics loguru pydantic-settings
python -m detector.train
```

This saves `detector/best.pt`.

### 4. Launch all services

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| EMQX Dashboard | http://localhost:18083 (admin / public) |
| MongoDB | mongodb://localhost:27017 |

### 5. Use the dashboard

1. Open **http://localhost:8000**.
2. Click **Upload Image** and select a JPG/PNG of a machine part.
3. Click **Run OCR**.
4. View the detected text, confidence, and annotated bounding box.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload image → run OCR pipeline |
| `GET` | `/api/latest` | Most recent OCR result |
| `GET` | `/api/history?limit=50` | Paginated result history |
| `DELETE` | `/api/history` | Clear all history |
| `GET` | `/api/search/{filename}` | Search by filename |
| `GET` | `/api/health` | Service health check |

---

## Development (no Docker)

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start MongoDB and EMQX manually (or update .env to point to existing instances)

# Run backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## MQTT Topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `industrial/ocr/image` | app → broker | Reserved for camera frame metadata (Phase 2) |
| `industrial/ocr/results` | app → broker | OCR result JSON |
| `industrial/ocr/status` | app → broker | Client online/offline status |

QoS level: **1** (at-least-once delivery)

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Core OCR pipeline + dashboard | ✅ Complete |
| 2 | Live camera integration (OpenCV VideoCapture) | 🔲 Planned |
| 3 | WebSocket real-time dashboard updates | 🔲 Planned |

---

## Preprocessing Pipeline

Each detected ROI passes through these OpenCV stages before PaddleOCR:

1. Grayscale conversion
2. CLAHE (contrast limited adaptive histogram equalisation)
3. Gaussian blur (noise reduction)
4. Non-local means denoising
5. Contrast enhancement (linear stretch)
6. Unsharp-mask sharpening
7. Adaptive thresholding (binarisation)

---

