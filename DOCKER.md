# Run NETRIKA with Docker

Install and start Docker Desktop with Linux containers enabled. Then run these commands from this project folder.

If `.env` does not exist, copy `.env.example` to `.env`. Set a random `SECRET_KEY` and your teacher credentials. Generate a secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Configure SMTP credentials for student password reset emails.

```powershell
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

Open http://localhost:8080. The initial build downloads Python, CPU PyTorch, OpenCV, and MongoDB and may take several minutes. No GPU is required.

```powershell
docker compose logs -f app
docker compose down
```

Stopping with `docker compose down` preserves database data in the named volume. Do not add `--volumes` unless you intend to delete that data.

The container database starts empty. Your existing local MongoDB data is not automatically copied; local accounts, face images, and reports remain in your local database. Compose overrides `MONGODB_URI` to use the `mongo` container. MongoDB is accessible only on the Docker network, with no host port published. The app is bound to localhost on port 8080, leaving the existing app on port 8000 alone.

The image includes only the app, frontend assets, and face models. `.env`, logs, unrelated documents, and local database files are excluded from the build context. One app worker is used because enrollment and inference locks are in-process.

Docker is a way to run the application, not a public hosting service. This setup runs on your computer and needs it to stay on. For a public demonstration, an HTTPS tunnel can point to http://localhost:8080. Before exposing it, use unique teacher credentials and a strong session secret, set `COOKIE_SECURE=true`, and set `APP_BASE_URL` to the public HTTPS address so reset links work. Restart the app after environment changes with `docker compose up -d --force-recreate app`.

Verified locally with Docker Desktop: image build completed, both containers reported healthy, the homepage returned HTTP 200, MongoDB ping succeeded, YOLO inference ran, and SFace produced a 128-value feature. These smoke checks do not measure recognition accuracy on real crowd footage.
