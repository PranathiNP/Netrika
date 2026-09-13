# NETRIKA database and email setup

## How the system works

Registration follows this path:

1. The visitor fills in the registration form in `index.html`.
2. JavaScript converts the fields to JSON and sends them to `/api/register`.
3. FastAPI validates the input and hashes the password.
4. MongoDB stores the name, email, password hash, and role.
5. FastAPI sends a welcome email through the configured SMTP server.

Login follows a similar path. FastAPI finds the user by email and role, compares the entered password with the saved hash, and creates a signed login session when they match.

## Why passwords are hashed

The database never stores the readable password. `generate_password_hash()` creates a one-way value. During login, `check_password_hash()` safely checks whether the entered password matches it.

The welcome email includes the login email and role, but not the password. Email is not a secure password-storage system. Users should use the password they created and later use a password-reset link if they forget it.

## 1. Install the required software

Install Python 3 and MongoDB Community Server. MongoDB Compass is optional but makes it easier to inspect records.

Open a terminal in this project and create a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Create the database

Start MongoDB. The application automatically creates the `netrika` database, the `users` collection, and a unique index on `email` when the first account is registered or a login is attempted.

The important document fields are:

- `full_name`: the user's display name
- `email`: the unique login name
- `password_hash`: the protected password value
- `role`: either `student` or `teacher`
- `created_at`: the registration time

## 3. Configure private settings

Copy `.env.example` to a new file named `.env`. Do not rename the example itself.

Set `MONGODB_URI` to your local MongoDB or MongoDB Atlas connection string and set `MONGODB_DB` to the database name. Replace `SECRET_KEY` with a long random value.

For Gmail SMTP:

1. Enable two-step verification on the sender's Google account.
2. Create a Google App Password.
3. Put the email address in `SMTP_USER` and `SMTP_FROM`.
4. Put the 16-character App Password in `SMTP_PASSWORD`.

The example values such as `your-email@gmail.com` and `your-app-password` do not send email. Replace them in `.env`, then restart FastAPI. Do not include spaces in the Google App Password.

Never commit `.env`. It is listed in `.gitignore` because it contains secrets.

## 4. Start the application

With the virtual environment active, run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser. FastAPI's interactive API documentation is available at `http://127.0.0.1:8000/docs`. Do not open `index.html` directly because direct file mode cannot reach the API routes correctly.

## 5. Test the flow

1. Open Register and create a student account. Teacher registration is disabled.
2. Enter a name, email, and password of at least eight characters.
3. Confirm that a document appears in the `netrika.users` collection.
4. Check that the welcome email arrives.
5. Student login uses the saved email and password. Teacher login uses the single administrator account configured with `TEACHER_USERNAME` and `TEACHER_PASSWORD` in `.env`.

After a successful student registration and welcome-email delivery, the browser displays a three-second confirmation message. Successful student and teacher logins open their separate protected dashboard pages.

## Student face registration

From the student dashboard, open the profile menu and select **Face Registration**. After the animated instructions, the browser requests camera permission and guides the student through front, left, right, and smiling captures. A face-specific YOLOv8 model checks that exactly one face is visible before accepting each frame.

Accepted JPEG frames are stored through MongoDB GridFS in `face_frames.files` and `face_frames.chunks`. Each file includes the authenticated user ID, requested action, YOLO bounding box, and capture time in its metadata. Biometric images are sensitive data; production deployments should add explicit consent, retention/deletion controls, encryption, and restricted administrative access.

## Teacher face recognition

The protected teacher portal accepts a photo or a video no longer than five seconds. YOLO locates faces and OpenCV SFace compares them with the students' registered face frames. A matched student is annotated with a white box and name; an unmatched face is annotated with a red box and `Unknown`. Annotated results are stored in the MongoDB GridFS collections `recognition_results.files` and `recognition_results.chunks`.

You can inspect accounts in `mongosh` with:

```javascript
use netrika
db.users.find({}, { password_hash: 0 })
```

Do not select or display `password_hash` in user-facing pages.

## Future extension

Keep the shared login information in `users`. When the project needs role-specific information, add role-specific profile collections linked to the user's `_id`. This avoids duplicated authentication code while allowing fields such as student USN, semester, teacher employee ID, and department.
