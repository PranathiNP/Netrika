# Attendance flow

## Dashboard and password recovery updates

Both dashboards open on attendance analytics, refreshed every 15 seconds while the tab is visible. Use the profile menu icons to open face tools or attendance reports. The teacher's present count refers to the latest session; the student's count is sessions attended. Attendance rate covers all recorded sessions available to that user.

Camera updates start at 0% and advance to 25%, 50%, 75%, and 100% as Front, Left, Right, and Smile frames are saved. Poses are user-guided; the detector validates one face, not the expression. Old images remain active until the smile frame completes the new set. Completion replaces all previous camera and video images. Closing capture removes staged images; the next capture also cleans up abandoned staged images. A successful video upload likewise replaces the previous image set.

Forgot password is available only on the student login form. Reset links expire after 15 minutes and can be used once; requests for the same account are limited to one per minute. Configure real `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, and optionally `SMTP_FROM` values in `.env`. Placeholder credentials cannot deliver email. Set `APP_BASE_URL` to the app's reachable address (defaults to `http://127.0.0.1:8000`). Teacher password recovery is disabled.

Run `.venv\Scripts\python.exe -m unittest test_attendance test_dashboard_updates -v` for database-backed regression checks. Email is mocked in tests; no real email is sent.

Start MongoDB and run `.venv\Scripts\python.exe app.py`, then open http://127.0.0.1:8000.

1. Register a student account and log in. Upload a video up to 15 seconds / 30 MB using **Save face images**, or capture four camera poses from the profile menu. Video enrollment samples up to 12 frames and requires at least four frames containing exactly one face. Images are stored in MongoDB GridFS, associated with the logged-in student ID.
2. Log in as a teacher using the credentials configured in `.env`. Upload a crowd photo or a video up to five seconds / 30 MB and select **Identify faces**.
3. Matched faces receive names; unmatched detected faces receive **Unknown**. Each successful upload saves a separate attendance session, with matched student IDs marked **Present** and all other registered students marked **Not detected**. Repeated appearances within one video mark a student only once. Unknown detections can repeat across frames and are not a count of unique people.
4. The teacher report refreshes after recognition and persists across reloads. Students use **Refresh report** to see their own attendance. The API provides no student attendance editing route and never accepts a client-selected student ID for viewing reports.

There is currently one global student roster; courses and scheduled class sessions are not modeled. Every upload creates a new session. Not detected does not prove absence.

Detection supports crowds larger than 40 faces (300 detection limit, 1280 inference size). Actual identification depends on clear enrollment images and visible faces in the crowd. The threshold defaults to 0.42 and can be configured with `FACE_MATCH_THRESHOLD`; validate it using representative images before relying on attendance. Existing SFace extraction uses resized face crops, without landmark alignment. No real 40-person accuracy benchmark has been run.

Annotated videos use MPEG-4 encoding; browser playback depends on codec support. A download link is included for viewing the result in a compatible player.

Run `.venv\Scripts\python.exe -m unittest test_attendance -v` for regression tests. MongoDB must be available. Tests use and delete a unique `netrika_test_*` database. Inference is mocked for deterministic identity tests; image storage, video decoding, session authorization and attendance persistence use the real application and database.
