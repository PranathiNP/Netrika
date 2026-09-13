import base64
from datetime import datetime, timedelta, timezone
import unittest
from uuid import uuid4
from urllib.parse import urlsplit, parse_qs
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from gridfs import GridFS
from werkzeug.security import generate_password_hash
import app


class DashboardUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.db = app.mongo_client['netrika_test_' + uuid4().hex]
        self.addCleanup(lambda: app.mongo_client.drop_database(self.db.name))
        for name, value in [('database', self.db), ('face_frames', GridFS(self.db, collection='face_frames'))]:
            patcher = patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.user_id = str(self.db.users.insert_one({'full_name': 'Student', 'email': 'student@test.local', 'role': 'student', 'password_hash': generate_password_hash('old-password')}).inserted_id)
        self.client = TestClient(app.app)

    def login(self):
        self.assertEqual(self.client.post('/api/login', json={'role': 'student', 'email': 'student@test.local', 'password': 'old-password'}).status_code, 200)

    def test_password_reset_single_use_expiry_and_new_login(self):
        with patch('password_reset.send_reset_email') as mail:
            response = self.client.post('/api/forgot-password', json={'email': 'student@test.local'})
            self.assertEqual(response.status_code, 200)
            token = parse_qs(urlsplit(mail.call_args.args[1]).fragment)['token'][0]
            self.client.post('/api/forgot-password', json={'email': 'student@test.local'})
            self.assertEqual(mail.call_count, 1)
            self.client.post('/api/forgot-password', json={'email': 'unknown@test.local'})
            self.assertEqual(mail.call_count, 1)
        payload = {'token': token, 'password': 'new-password'}
        self.assertEqual(self.client.post('/api/reset-password', json=payload).status_code, 200)
        self.assertEqual(self.client.post('/api/reset-password', json=payload).status_code, 400)
        self.assertEqual(self.client.post('/api/login', json={'email': 'student@test.local', 'role': 'student', 'password': 'old-password'}).status_code, 401)
        self.assertEqual(self.client.post('/api/login', json={'email': 'student@test.local', 'role': 'student', 'password': 'new-password'}).status_code, 200)
        self.db.users.update_many({}, {'$unset': {'reset_requested_at': ''}})
        with patch('password_reset.send_reset_email') as mail:
            self.client.post('/api/forgot-password', json={'email': 'student@test.local'})
            token = parse_qs(urlsplit(mail.call_args.args[1]).fragment)['token'][0]
        self.db.users.update_many({}, {'$set': {'reset_expires': datetime.now(timezone.utc) - timedelta(seconds=1)}})
        self.assertEqual(self.client.post('/api/reset-password', json={'token': token, 'password': 'another-password'}).status_code, 400)

    def test_teacher_recovery_disabled(self):
        with patch('password_reset.send_reset_email') as mail:
            self.assertEqual(self.client.post('/api/forgot-password', json={'email': 'teacher@test.local', 'role': 'teacher'}).status_code, 400)
            self.assertEqual(self.client.post('/api/reset-password', json={'token': 'x' * 32, 'role': 'teacher', 'password': 'teacher-new-password'}).status_code, 400)
            mail.assert_not_called()
        self.assertEqual(self.db.auth_settings.count_documents({}), 0)

    def test_camera_replaces_all_old_images_only_after_smile(self):
        self.login()
        original = app.face_frames.put(b'old-image', metadata={'user_id': self.user_id, 'action': 'video'})
        _, encoded = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        image = base64.b64encode(encoded).decode()
        boxes = Mock()
        boxes.__len__ = Mock(return_value=1)
        boxes.xyxy = np.array([[4., 4., 50., 50.]])
        detector = Mock()
        detector.predict.return_value = [Mock(boxes=boxes)]
        capture_id = self.client.post('/api/face-registration/start').json()['capture_id']
        with patch.object(app, 'get_face_model', return_value=detector):
            self.assertEqual(self.client.post('/api/face-registration/frame', json={'capture_id': capture_id, 'action': 'smile', 'image': image}).status_code, 409)
            for index, action in enumerate(['front', 'left', 'right', 'smile']):
                response = self.client.post('/api/face-registration/frame', json={'capture_id': capture_id, 'action': action, 'image': image})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['progress'], (index + 1) * 25)
                self.assertEqual(app.face_frames.exists(original), action != 'smile')
        self.assertEqual(self.db['face_frames.files'].count_documents({}), 4)
        self.client.delete('/api/face-registration/capture/' + capture_id)
        self.assertEqual(self.db['face_frames.files'].count_documents({}), 4)
        capture_id = self.client.post('/api/face-registration/start').json()['capture_id']
        with patch.object(app, 'get_face_model', return_value=detector):
            self.client.post('/api/face-registration/frame', json={'capture_id': capture_id, 'action': 'front', 'image': image})
        self.assertEqual(self.client.get('/api/face-registration').json()['saved_frames'], 4)
        self.client.delete('/api/face-registration/capture/' + capture_id)
        self.assertEqual(self.db['face_frames.files'].count_documents({}), 4)


if __name__ == '__main__':
    unittest.main()
