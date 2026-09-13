FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    FASTAPI_RELOAD=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py password_reset.py ./
COPY index.html student.html teacher.html password_reset.html ./
COPY style.css dashboard.css attendance.js ./
COPY Background.jpg dashboard-clouds.png ./
COPY yolov8n-face.pt face_recognition_sface.onnx ./

RUN useradd --create-home --uid 10001 netrika && chown -R netrika:netrika /app
USER netrika
EXPOSE 8000

# One worker preserves the application's in-process model/enrollment locks.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
