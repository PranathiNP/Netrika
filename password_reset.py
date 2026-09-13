"""Email password recovery for students."""
import hashlib
import os
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from pymongo.errors import PyMongoError
from werkzeug.security import generate_password_hash


class ForgotPassword(BaseModel):
    email: str
    role: str = 'student'


class ResetPassword(BaseModel):
    token: str
    password: str
    role: str = 'student'


def send_reset_email(email, link):
    host, user, password = (os.getenv(key, '') for key in ('SMTP_HOST', 'SMTP_USER', 'SMTP_PASSWORD'))
    if not all((host, user, password)) or user == 'your-email@gmail.com' or password == 'your-app-password':
        raise RuntimeError('Email delivery is not configured.')
    message = EmailMessage()
    message['Subject'] = 'Reset your NETRIKA password'
    message['From'] = os.getenv('SMTP_FROM', user)
    message['To'] = email
    message.set_content(f'Use this link to choose a new password:\n\n{link}\n\nThis link expires in 15 minutes and works once. If you did not request this, ignore this email.')
    with smtplib.SMTP(host, int(os.getenv('SMTP_PORT', '587')), timeout=15) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(message)


def install_routes(application, get_database, app_dir):
    @application.get('/forgot-password', include_in_schema=False)
    @application.get('/reset-password', include_in_schema=False)
    def recovery_page():
        return FileResponse(app_dir / 'password_reset.html', headers={'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store'})

    @application.post('/api/forgot-password')
    def forgot_password(data: ForgotPassword):
        email = data.email.strip().lower()
        if data.role != 'student' or '@' not in email or len(email) > 254:
            return JSONResponse({'message': 'Enter a valid email and role.'}, status_code=400)
        generic = {'message': 'If this email is registered, a reset link will be sent. Check your inbox and spam folder.'}
        db = get_database()
        try:
            collection = db['users']
            query = {'email': email, 'role': 'student'}
            now = datetime.now(timezone.utc)
            query['$or'] = [{'reset_requested_at': {'$lt': now - timedelta(seconds=60)}}, {'reset_requested_at': {'$exists': False}}]
            token = secrets.token_urlsafe(32)
            digest = hashlib.sha256(token.encode()).hexdigest()
            user = collection.find_one_and_update(query, {'$set': {'reset_hash': digest, 'reset_expires': now + timedelta(minutes=15), 'reset_requested_at': now}})
            if user:
                try:
                    base_url = os.getenv('APP_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
                    send_reset_email(email, f'{base_url}/reset-password#token={token}&role={data.role}')
                except (RuntimeError, ValueError, OSError, smtplib.SMTPException):
                    collection.update_one({'_id': user['_id'], 'reset_hash': digest}, {'$unset': {'reset_hash': '', 'reset_expires': '', 'reset_requested_at': ''}})
                    return JSONResponse({'message': 'Reset email could not be sent. Please contact the administrator to check email settings.'}, status_code=503)
            return generic
        except PyMongoError:
            return JSONResponse({'message': 'Password recovery is temporarily unavailable.'}, status_code=503)

    @application.post('/api/reset-password')
    def reset_password(data: ResetPassword):
        if data.role != 'student' or not 8 <= len(data.password) <= 256 or not 20 <= len(data.token) <= 128:
            return JSONResponse({'message': 'Use a valid reset link and a password of 8–256 characters.'}, status_code=400)
        try:
            collection = get_database()['users']
            query = {'reset_hash': hashlib.sha256(data.token.encode()).hexdigest(), 'reset_expires': {'$gt': datetime.now(timezone.utc)}}
            query['role'] = 'student'
            user = collection.find_one_and_update(query, {'$set': {'password_hash': generate_password_hash(data.password)}, '$unset': {'reset_hash': '', 'reset_expires': ''}})
            if not user:
                return JSONResponse({'message': 'This link has expired or was already used. Request a new reset email.'}, status_code=400)
            return {'message': 'Password updated. You can now log in with your new password.'}
        except PyMongoError:
            return JSONResponse({'message': 'Password recovery is temporarily unavailable.'}, status_code=503)
