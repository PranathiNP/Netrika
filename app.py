import logging
import os
import smtplib
import base64
import binascii
import tempfile
import math
import secrets
from functools import wraps
from threading import RLock
from hmac import compare_digest
from io import BytesIO
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from bson import ObjectId
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from pymongo import ASCENDING, MongoClient
from gridfs import GridFS
from pymongo.errors import DuplicateKeyError, PyMongoError
from starlette.middleware.sessions import SessionMiddleware
from werkzeug.security import check_password_hash, generate_password_hash


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("netrika")

app = FastAPI(title="NETRIKA API")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "change-this-before-production"),
    same_site="lax",
    https_only=os.getenv("COOKIE_SECURE", "false").lower() == "true",
)

app_dir = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(app_dir / ".ultralytics"))
mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
database = mongo_client[os.getenv("MONGODB_DB", "netrika")]
face_frames = GridFS(database, collection="face_frames")
recognition_results = GridFS(database, collection="recognition_results")
face_model = None
recognition_model = None
model_lock = RLock()
enrollment_lock = RLock()


def serialized_enrollment(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with enrollment_lock:
            return function(*args, **kwargs)
    return wrapped


class AuthPayload(BaseModel):
    email: str = ""
    username: str = ""
    password: str = ""
    role: str = ""


class RegistrationPayload(AuthPayload):
    name: str = ""


class FaceFramePayload(BaseModel):
    image: str
    action: str
    capture_id: str = ""


def users_collection():
    users = database["users"]
    users.create_index([("email", ASCENDING)], unique=True, name="uq_users_email")
    return users


def get_face_model():
    global face_model
    if face_model is None:
        from ultralytics import YOLO

        model_path = Path(os.getenv("YOLO_FACE_MODEL", "yolov8n-face.pt"))
        if not model_path.is_absolute():
            model_path = app_dir / model_path
        face_model = YOLO(str(model_path))
    return face_model


def get_recognition_model():
    global recognition_model
    if recognition_model is None:
        import cv2

        model_path = Path(os.getenv("SFACE_MODEL", "face_recognition_sface.onnx"))
        if not model_path.is_absolute():
            model_path = app_dir / model_path
        recognition_model = cv2.FaceRecognizerSF.create(str(model_path), "")
    return recognition_model


def face_feature(image, box):
    import cv2
    import numpy as np

    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    padding_x = int((x2 - x1) * 0.18)
    padding_y = int((y2 - y1) * 0.18)
    crop = image[max(0, y1-padding_y):min(height, y2+padding_y), max(0, x1-padding_x):min(width, x2+padding_x)]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (112, 112))
    with model_lock:
        feature = get_recognition_model().feature(crop).flatten().astype("float32")
    norm = np.linalg.norm(feature)
    return feature / norm if norm else None


def known_student_features():
    import cv2
    import numpy as np

    known = []
    stored_files = database["face_frames.files"]
    for user in users_collection().find({"role": "student"}, {"full_name": 1}):
        features = []
        for stored in stored_files.find({"metadata.user_id": str(user["_id"]), "metadata.pending": {"$ne": True}}):
            image_bytes = face_frames.get(stored["_id"]).read()
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            feature = face_feature(image, stored["metadata"]["face_box"])
            if feature is not None:
                features.append(feature)
        if features:
            average = np.mean(features, axis=0)
            norm = np.linalg.norm(average)
            if norm > 0:
                known.append((str(user["_id"]), user["full_name"], average / norm))
    return known


def annotate_faces(image, known):
    import cv2
    import numpy as np

    with model_lock:
        result = get_face_model().predict(image, conf=0.45, imgsz=1280, max_det=300, verbose=False)[0]
    recognized = []
    source = image.copy()
    for raw_box in result.boxes.xyxy.tolist():
        box = [int(value) for value in raw_box]
        feature = face_feature(source, box)
        student_id = None
        name = "Unknown"
        best_score = -1.0
        if feature is not None:
            for known_id, student_name, student_feature in known:
                score = float(np.dot(feature, student_feature))
                if score > best_score:
                    student_id, name, best_score = known_id, student_name, score
        if best_score < float(os.getenv("FACE_MATCH_THRESHOLD", "0.42")):
            name = "Unknown"
            student_id = None
        color = (255, 255, 255) if name != "Unknown" else (0, 0, 255)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label_y = max(25, y1 - 10)
        cv2.putText(image, name, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        recognized.append({"student_id": student_id, "name": name})
    return image, recognized


def send_welcome_email(name: str, email: str, role: str) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    placeholder_values = {"your-email@gmail.com", "your-app-password"}
    if (
        not all((smtp_host, smtp_user, smtp_password))
        or smtp_user in placeholder_values
        or smtp_password in placeholder_values
    ):
        raise RuntimeError("SMTP credentials are not configured in .env")

    message = EmailMessage()
    message["Subject"] = "Welcome to NETRIKA"
    message["From"] = os.getenv("SMTP_FROM", smtp_user)
    message["To"] = email
    message.set_content(
        f"Hello {name},\n\n"
        f"Thank you for registering with NETRIKA as a {role}.\n"
        f"Your login email is: {email}\n\n"
        "For your security, your password is not included in this email. "
        "Please use the password you created during registration.\n\n"
        "Regards,\nNETRIKA"
    )

    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(smtp_host, port, timeout=15) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(message)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(app_dir / "index.html")


@app.get("/attendance.js", include_in_schema=False)
def attendance_script():
    return FileResponse(app_dir / "attendance.js", media_type="application/javascript")


@app.get("/style.css", include_in_schema=False)
def stylesheet():
    return FileResponse(app_dir / "style.css", media_type="text/css")


@app.get("/Background.jpg", include_in_schema=False)
def background_image():
    return FileResponse(app_dir / "Background.jpg", media_type="image/jpeg")


@app.get("/dashboard-clouds.png", include_in_schema=False)
def dashboard_background():
    return FileResponse(app_dir / "dashboard-clouds.png", media_type="image/png")


@app.get("/dashboard.css", include_in_schema=False)
def dashboard_stylesheet():
    return FileResponse(app_dir / "dashboard.css", media_type="text/css")


@app.get("/student", include_in_schema=False)
def student_page(request: Request):
    if request.session.get("role") != "student":
        return JSONResponse({"message": "Student login required."}, status_code=401)
    return FileResponse(app_dir / "student.html")


@app.get("/api/me")
def current_user(request: Request):
    if not request.session.get("user_id"):
        return JSONResponse({"message": "Login required."}, status_code=401)
    return {
        "name": request.session.get("name", "User"),
        "role": request.session.get("role"),
    }


@app.post("/api/face-registration/start")
@serialized_enrollment
def start_face_capture(request: Request):
    if request.session.get("role") != "student":
        return JSONResponse({"message": "Student login required."}, status_code=401)
    try:
        user_id = request.session["user_id"]
        for old in database["face_frames.files"].find({"metadata.user_id": user_id, "metadata.pending": True}, {"_id": 1}):
            face_frames.delete(old["_id"])
        database["face_capture_sessions"].delete_many({"user_id": user_id})
        capture_id = secrets.token_urlsafe(24)
        database["face_capture_sessions"].insert_one({"_id": capture_id, "user_id": user_id, "next_action": "front"})
        return {"capture_id": capture_id, "progress": 0}
    except PyMongoError:
        return JSONResponse({"message": "Could not start face capture."}, status_code=503)


@app.delete("/api/face-registration/capture/{capture_id}")
@serialized_enrollment
def cancel_face_capture(capture_id: str, request: Request):
    if request.session.get("role") != "student":
        return JSONResponse({"message": "Student login required."}, status_code=401)
    try:
        user_id = request.session["user_id"]
        for old in database["face_frames.files"].find({"metadata.user_id": user_id, "metadata.capture_id": capture_id, "metadata.pending": True}, {"_id": 1}):
            face_frames.delete(old["_id"])
        database["face_capture_sessions"].delete_one({"_id": capture_id, "user_id": user_id})
        return {"message": "Capture closed."}
    except PyMongoError:
        return JSONResponse({"message": "Capture cleanup could not be completed."}, status_code=503)


@app.post("/api/face-registration/frame")
@serialized_enrollment
def save_face_frame(data: FaceFramePayload, request: Request):
    if request.session.get("role") != "student":
        return JSONResponse({"message": "Student login required."}, status_code=401)
    if data.action not in {"front", "left", "right", "smile"}:
        return JSONResponse({"message": "Invalid face action."}, status_code=400)

    try:
        user_id = request.session["user_id"]
        capture = database["face_capture_sessions"].find_one({"_id": data.capture_id, "user_id": user_id})
        if not capture or capture["next_action"] != data.action:
            return JSONResponse({"message": "Start a new capture and follow the pose order."}, status_code=409)
        encoded_image = data.image.split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded_image, validate=True)
        if len(image_bytes) > 2_000_000:
            return JSONResponse({"message": "Frame is too large."}, status_code=413)

        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse({"message": "Invalid camera frame."}, status_code=400)
        with model_lock:
            result = get_face_model().predict(image, conf=0.45, verbose=False)[0]
        if len(result.boxes) != 1:
            return JSONResponse(
                {"message": "Keep exactly one face inside the circle."}, status_code=422
            )
        box = [round(float(value), 2) for value in result.boxes.xyxy[0].tolist()]

        user_id = request.session["user_id"]
        stored_frames = database["face_frames.files"]
        existing = list(stored_frames.find({"metadata.user_id": user_id, "metadata.capture_id": data.capture_id, "metadata.action": data.action}, {"_id": 1}))
        face_frames.put(image_bytes, filename=f"{user_id}-{data.action}.jpg", content_type="image/jpeg",
            metadata={"user_id": user_id, "action": data.action, "face_box": box,
                      "pending": True, "capture_id": data.capture_id, "captured_at": datetime.now(timezone.utc)})
        for old in existing:
            face_frames.delete(old["_id"])
        actions = ["front", "left", "right", "smile"]
        completed = actions.index(data.action) + 1
        if completed == 4:
            stored_frames.update_many({"metadata.user_id": user_id, "metadata.capture_id": data.capture_id}, {"$set": {"metadata.pending": False}})
            for old in stored_frames.find({"metadata.user_id": user_id, "metadata.capture_id": {"$ne": data.capture_id}}, {"_id": 1}):
                face_frames.delete(old["_id"])
            database["face_capture_sessions"].delete_one({"_id": data.capture_id})
        else:
            database["face_capture_sessions"].update_one({"_id": data.capture_id}, {"$set": {"next_action": actions[completed]}})
        return {
            "message": f"{data.action.title()} frame saved.",
            "progress": completed * 25,
            "face_box": box,
        }
    except (ValueError, binascii.Error):
        return JSONResponse({"message": "Invalid image data."}, status_code=400)
    except (ImportError, FileNotFoundError) as error:
        logger.exception("YOLO face detector is unavailable")
        return JSONResponse(
            {"message": f"YOLO face detector is unavailable: {error}"}, status_code=503
        )
    except PyMongoError:
        logger.exception("Face frame storage error")
        return JSONResponse({"message": "Could not store the face frame."}, status_code=500)


@app.get("/teacher", include_in_schema=False)
def teacher_page(request: Request):
    if request.session.get("role") != "teacher":
        return JSONResponse({"message": "Teacher login required."}, status_code=401)
    return FileResponse(app_dir / "teacher.html")


@app.post("/api/recognize")
def recognize_faces(request: Request, media: UploadFile = File(...)):
    if request.session.get("role") != "teacher":
        return JSONResponse({"message": "Teacher login required."}, status_code=401)

    content_type = media.content_type or ""
    if not (content_type.startswith("image/") or content_type.startswith("video/")):
        return JSONResponse({"message": "Upload a photo or video."}, status_code=400)
    media_bytes = media.file.read(30_000_001)
    if not media_bytes or len(media_bytes) > 30_000_000:
        return JSONResponse({"message": "The file must be smaller than 30 MB."}, status_code=413)

    try:
        import cv2
        import numpy as np

        known = known_student_features()

        names = set()
        present_ids = set()
        unknown_detections = 0
        if content_type.startswith("image/"):
            image = cv2.imdecode(np.frombuffer(media_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return JSONResponse({"message": "The photo could not be read."}, status_code=400)
            annotated, frame_names = annotate_faces(image, known)
            names.update(face["name"] for face in frame_names)
            present_ids.update(face["student_id"] for face in frame_names if face["student_id"])
            unknown_detections += sum(face["student_id"] is None for face in frame_names)
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Could not encode the annotated photo")
            result_bytes = encoded.tobytes()
            result_type = "image/jpeg"
            filename = "recognized-photo.jpg"
        else:
            suffix = Path(media.filename or "video.mp4").suffix or ".mp4"
            with tempfile.TemporaryDirectory(dir=app_dir) as temp_dir:
                input_path = Path(temp_dir) / f"input{suffix}"
                output_path = Path(temp_dir) / "recognized-video.mp4"
                input_path.write_bytes(media_bytes)
                capture = cv2.VideoCapture(str(input_path))
                writer = None
                try:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0 or count / fps > 5.25:
                        return JSONResponse({"message": "Upload a readable video of five seconds or shorter."}, status_code=400)
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Video encoder is unavailable")
                    processed = 0
                    while True:
                        available, frame = capture.read()
                        if not available:
                            break
                        processed += 1
                        if processed > math.ceil(fps * 5.25):
                            return JSONResponse({"message": "Video must be five seconds or shorter."}, status_code=400)
                        annotated, frame_names = annotate_faces(frame, known)
                        names.update(face["name"] for face in frame_names)
                        present_ids.update(face["student_id"] for face in frame_names if face["student_id"])
                        unknown_detections += sum(face["student_id"] is None for face in frame_names)
                        writer.write(annotated)
                    if not processed:
                        raise RuntimeError("No video frames could be decoded")
                finally:
                    capture.release()
                    if writer is not None:
                        writer.release()
                result_bytes = output_path.read_bytes()
            result_type = "video/mp4"
            filename = "recognized-video.mp4"

        result_id = recognition_results.put(
            result_bytes,
            filename=filename,
            content_type=result_type,
            metadata={
                "teacher_id": request.session["user_id"],
                "identified_names": sorted(names),
                "created_at": datetime.now(timezone.utc),
            },
        )
        report = {
            "teacher_id": request.session["user_id"],
            "created_at": datetime.now(timezone.utc),
            "source_name": Path(media.filename or "Upload").name,
            "result_id": str(result_id),
            "unknown_detections": unknown_detections,
            "students": [
                {"student_id": str(user["_id"]), "name": user["full_name"],
                 "status": "Present" if str(user["_id"]) in present_ids else "Not detected"}
                for user in users_collection().find({"role": "student"}, {"full_name": 1})
            ],
        }
        try:
            database["attendance_sessions"].insert_one(report)
        except PyMongoError:
            recognition_results.delete(result_id)
            raise
        return {
            "message": "Recognition complete. Attendance saved.",
            "attendance": serialize_report(report),
            "result_url": f"/api/recognition/result/{result_id}",
            "media_type": result_type,
            "names": sorted(names),
        }
    except (ImportError, FileNotFoundError, RuntimeError) as error:
        logger.exception("Face recognition service error")
        return JSONResponse({"message": f"Recognition failed: {error}"}, status_code=503)
    except PyMongoError:
        logger.exception("Recognition database error")
        return JSONResponse({"message": "Recognition data could not be loaded."}, status_code=500)
    except Exception:
        logger.exception("Unexpected face recognition error")
        return JSONResponse({"message": "Face recognition could not be completed."}, status_code=500)


@app.get("/api/recognition/result/{result_id}")
def recognition_result(result_id: str, request: Request):
    if request.session.get("role") != "teacher":
        return JSONResponse({"message": "Teacher login required."}, status_code=401)
    try:
        stored = recognition_results.get(ObjectId(result_id))
        if stored.metadata.get("teacher_id") != request.session.get("user_id"):
            return JSONResponse({"message": "Recognition result was not found."}, status_code=404)
        return StreamingResponse(
            BytesIO(stored.read()),
            media_type=stored.content_type,
            headers={"Content-Disposition": f'inline; filename="{stored.filename}"'},
        )
    except Exception:
        return JSONResponse({"message": "Recognition result was not found."}, status_code=404)


def serialize_report(report, student_id=None):
    students = report["students"]
    if student_id is not None:
        students = [row for row in students if row["student_id"] == student_id]
    created = report["created_at"].replace(tzinfo=timezone.utc)
    return {"id": str(report["_id"]), "created_at": created.isoformat(),
            "source_name": report["source_name"] if student_id is None else "Attendance session",
            "students": students,
            "present_count": sum(row["status"] == "Present" for row in students),
            "unknown_detections": report.get("unknown_detections", 0) if student_id is None else 0}


@app.get("/api/attendance")
def attendance(request: Request):
    role = request.session.get("role")
    user_id = request.session.get("user_id")
    if not user_id or role not in {"teacher", "student"}:
        return JSONResponse({"message": "Login required."}, status_code=401)
    query = {"teacher_id": user_id} if role == "teacher" else {"students.student_id": user_id}
    try:
        reports = database["attendance_sessions"].find(query).sort("created_at", -1)
        return {"reports": [serialize_report(report, user_id if role == "student" else None) for report in reports]}
    except PyMongoError:
        return JSONResponse({"message": "Attendance could not be loaded."}, status_code=503)


@app.get("/api/face-registration")
def registration_status(request: Request):
    if request.session.get("role") != "student":
        return JSONResponse({"message": "Student login required."}, status_code=401)
    try:
        count = database["face_frames.files"].count_documents({"metadata.user_id": request.session["user_id"], "metadata.pending": {"$ne": True}})
        return {"saved_frames": count}
    except PyMongoError:
        return JSONResponse({"message": "Registration status could not be loaded."}, status_code=503)


@app.post("/api/face-registration/video")
@serialized_enrollment
def register_video(request: Request, media: UploadFile = File(...)):
    if request.session.get("role") != "student":
        return JSONResponse({"message": "Student login required."}, status_code=401)
    if not (media.content_type or "").startswith("video/"):
        return JSONResponse({"message": "Upload a video."}, status_code=400)
    content = media.file.read(30_000_001)
    if not content or len(content) > 30_000_000:
        return JSONResponse({"message": "Upload a nonempty video smaller than 30 MB."}, status_code=413)
    try:
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("input" + (Path(media.filename or "video.mp4").suffix or ".mp4"))
            path.write_bytes(content)
            capture = cv2.VideoCapture(str(path))
            frames = []
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0 or count / fps > 15.25:
                    return JSONResponse({"message": "Upload a readable video up to 15 seconds."}, status_code=400)
                for index in range(12):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index * max(0, count - 1) / 12))
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    with model_lock:
                        detected = get_face_model().predict(frame, conf=0.45, verbose=False)[0]
                    if len(detected.boxes) != 1:
                        continue
                    box = detected.boxes.xyxy[0].tolist()
                    if face_feature(frame, box) is None:
                        continue
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if ok:
                        frames.append((encoded.tobytes(), box))
            finally:
                capture.release()
        if len(frames) < 4:
            return JSONResponse({"message": "Need at least four clear frames with exactly one face. Try a longer, well-lit video."}, status_code=422)
        user_id = request.session["user_id"]
        old_ids = [item["_id"] for item in database["face_frames.files"].find({"metadata.user_id": user_id}, {"_id": 1})]
        new_ids = []
        try:
            for index, (image, box) in enumerate(frames):
                new_ids.append(face_frames.put(image, filename=f"{user_id}-video-{index}.jpg", content_type="image/jpeg",
                    metadata={"user_id": user_id, "action": "video", "face_box": box, "captured_at": datetime.now(timezone.utc)}))
        except Exception:
            for file_id in new_ids:
                face_frames.delete(file_id)
            raise
        for file_id in old_ids:
            face_frames.delete(file_id)
        database["face_capture_sessions"].delete_many({"user_id": user_id})
        return {"message": f"Saved {len(frames)} face images to the database.", "saved_frames": len(frames)}
    except PyMongoError:
        logger.exception("Video registration database error")
        return JSONResponse({"message": "Face images could not be saved."}, status_code=503)
    except Exception:
        logger.exception("Video registration failed")
        return JSONResponse({"message": "Video registration failed. Check the video and face models."}, status_code=503)


@app.post("/api/register")
def register(data: RegistrationPayload):
    name = data.name.strip()
    email = data.email.strip().lower()
    password = data.password
    role = "student"

    if not name or not email or not password:
        return JSONResponse(
            {"message": "Please complete every registration field."}, status_code=400
        )
    if len(password) < 8:
        return JSONResponse(
            {"message": "Password must contain at least 8 characters."}, status_code=400
        )

    try:
        users_collection().insert_one(
            {
                "full_name": name,
                "email": email,
                "password_hash": generate_password_hash(password),
                "role": role,
                "created_at": datetime.now(timezone.utc),
            }
        )
    except DuplicateKeyError:
        return JSONResponse(
            {"message": "An account with this email already exists."}, status_code=409
        )
    except PyMongoError:
        logger.exception("Registration database error")
        return JSONResponse(
            {"message": "Registration could not be completed."}, status_code=500
        )

    email_sent = True
    try:
        send_welcome_email(name, email, role)
    except RuntimeError as error:
        email_sent = False
        logger.warning("Welcome email skipped: %s", error)
    except (OSError, smtplib.SMTPException):
        email_sent = False
        logger.exception("Welcome email could not be sent")

    message = "Registration successful. You can now log in."
    if not email_sent:
        message += " The welcome email could not be sent; check the SMTP settings."
    return JSONResponse({"message": message, "email_sent": email_sent}, status_code=201)


@app.post("/api/login")
def login(data: AuthPayload, request: Request):
    email = data.email.strip().lower()
    username = data.username.strip()
    password = data.password
    role = data.role.strip().lower()

    if not password or role not in {"student", "teacher"} or (role == "student" and not email):
        return JSONResponse(
            {"message": "Email, password, and role are required."}, status_code=400
        )

    request.session.clear()
    if role == "teacher":
        valid_username = compare_digest(
            username.encode("utf-8"), os.getenv("TEACHER_USERNAME", "Admin123").encode("utf-8")
        )
        try:
            teacher_auth = database["auth_settings"].find_one({"_id": "teacher-admin"}) or {}
        except PyMongoError:
            return JSONResponse({"message": "Login is temporarily unavailable."}, status_code=503)
        if teacher_auth.get("password_hash"):
            valid_password = check_password_hash(teacher_auth["password_hash"], password)
        else:
            valid_password = compare_digest(password.encode("utf-8"), os.getenv("TEACHER_PASSWORD", "Admin@123").encode("utf-8"))
        if not valid_username or not valid_password:
            return JSONResponse(
                {"message": "Invalid teacher username or password."}, status_code=401
            )
        request.session["user_id"] = "teacher-admin"
        request.session["name"] = "Administrator"
        request.session["role"] = "teacher"
        return {
            "message": "Welcome, Administrator!",
            "role": "teacher",
            "redirect_url": "/teacher",
        }

    try:
        user = users_collection().find_one({"email": email, "role": "student"})
    except PyMongoError:
        logger.exception("Login database error")
        return JSONResponse(
            {"message": "Login could not be completed."}, status_code=500
        )

    if not user or not check_password_hash(user["password_hash"], password):
        return JSONResponse(
            {"message": "Invalid email or password."}, status_code=401
        )

    request.session["user_id"] = str(user["_id"])
    request.session["name"] = user["full_name"]
    request.session["role"] = "student"
    return {
        "message": f"Welcome, {user['full_name']}!",
        "role": "student",
        "redirect_url": "/student",
    }


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"message": "You have been logged out."}


from password_reset import install_routes

install_routes(app, lambda: database, app_dir)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=os.getenv("FASTAPI_RELOAD", "false").lower() == "true",
    )
