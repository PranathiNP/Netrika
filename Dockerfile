FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    FASTAPI_RELOAD=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --uid 10001 netrika && chown netrika:netrika /app

COPY --chown=netrika:netrika app.py password_reset.py face_matching.py media_delivery.py feature_store.py ./
COPY --chown=netrika:netrika index.html student.html teacher.html password_reset.html ./
COPY --chown=netrika:netrika style.css dashboard.css attendance.js ./
COPY --chown=netrika:netrika Background.jpg dashboard-clouds.png ./
COPY --chown=netrika:netrika yolov8n-face.pt face_recognition_sface.onnx face_detection_yunet.onnx ./

USER netrika
EXPOSE 8000

# One worker preserves the application's in-process model/enrollment locks.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
