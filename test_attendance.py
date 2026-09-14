"""Run with .venv/Scripts/python.exe -m unittest test_attendance -v.

Uses a uniquely named temporary MongoDB database; requires MongoDB running.
Face inference is mocked to test attendance and authorization deterministically.
"""
import unittest
import tempfile
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch, Mock

import cv2
import numpy as np
from fastapi.testclient import TestClient
from gridfs import GridFS
from werkzeug.security import generate_password_hash
import app


class AttendanceTests(unittest.TestCase):
    def test_crowd_40_and_unknown(self):
        boxes = Mock()
        boxes.xyxy.tolist.return_value = [[i * 5, 0, i * 5 + 4, 10] for i in range(40)]
        detector = Mock()
        detector.predict.return_value = [Mock(boxes=boxes)]
        features = list(np.eye(40))
        known = [(str(i), 'Student ' + str(i), np.eye(40)[i]) for i in range(39)]
        with patch.object(app, 'get_face_model', return_value=detector), patch.object(app, 'face_feature', side_effect=features):
            _, faces = app.annotate_faces(np.zeros((30, 210, 3), dtype=np.uint8), known)
        self.assertEqual(len(faces), 40)
        self.assertEqual(faces[-1], {'student_id': None, 'name': 'Unknown'})
        self.assertEqual(len({face['student_id'] for face in faces[:-1]}), 39)

    def test_persistence_and_student_read_only(self):
        db = app.mongo_client['netrika_test_' + uuid4().hex]
        try:
            with patch.object(app, 'database', db), patch.object(app, 'face_frames', GridFS(db, collection='face_frames')), patch.object(app, 'recognition_results', GridFS(db, collection='recognition_results')):
                ids = [str(db.users.insert_one({'full_name': 'Same Name', 'role': 'student', 'email': email, 'password_hash': generate_password_hash('test-pass-123')}).inserted_id) for email in ['one@test.local', 'two@test.local']]
                teacher = TestClient(app.app)
                with patch.dict('os.environ', {'TEACHER_USERNAME': 'test-teacher', 'TEACHER_PASSWORD': 'test-pass-123'}):
                    self.assertEqual(teacher.post('/api/login', json={'role': 'teacher', 'username': 'test-teacher', 'password': 'test-pass-123'}).status_code, 200)
                _, photo = cv2.imencode('.jpg', np.zeros((40, 40, 3), dtype=np.uint8))
                with patch.object(app, 'known_student_features', return_value=[]), patch.object(app, 'annotate_faces', side_effect=lambda image, known: (image, [{'student_id': ids[0], 'name': 'Same Name'}, {'student_id': None, 'name': 'Unknown'}])):
                    response = teacher.post('/api/recognize', files={'media': ('crowd.jpg', photo.tobytes(), 'image/jpeg')})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(db.attendance_sessions.count_documents({}), 1)
                report = teacher.get('/api/attendance').json()['reports'][0]
                self.assertEqual([r['status'] for r in report['students']], ['Present', 'Not detected'])
                self.assertEqual(report['unknown_detections'], 1)
                self.assertEqual(teacher.get(response.json()['result_url']).status_code, 200)
                student = TestClient(app.app)
                student.post('/api/login', json={'role': 'student', 'email': 'one@test.local', 'password': 'test-pass-123'})
                rows = student.get('/api/attendance?student_id=' + ids[1]).json()['reports'][0]['students']
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['student_id'], ids[0])
                with tempfile.TemporaryDirectory() as directory:
                    video_path = Path(directory) / 'enrollment.avi'
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
                    self.assertTrue(writer.isOpened())
                    for _ in range(20):
                        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
                    writer.release()
                    detector = Mock()
                    detector.predict.return_value = [Mock(boxes=np.array([[4., 4., 50., 50.]]))]
                    box = Mock()
                    box.__len__ = Mock(return_value=1)
                    box.xyxy = np.array([[4., 4., 50., 50.]])
                    detector.predict.return_value = [Mock(boxes=box)]
                    with patch.object(app, 'get_face_model', return_value=detector), patch.object(app, 'face_feature', return_value=np.array([1., 0.])):
                        enrolled = student.post('/api/face-registration/video', files={'media': ('enrollment.avi', video_path.read_bytes(), 'video/x-msvideo')})
                    self.assertEqual(enrolled.status_code, 200, enrolled.text)
                    self.assertEqual(db['face_frames.files'].count_documents({'metadata.user_id': ids[0]}), 12)
                    stored_image = app.face_frames.find_one({'metadata.user_id': ids[0]}).read()
                    self.assertIsNotNone(cv2.imdecode(np.frombuffer(stored_image, dtype=np.uint8), cv2.IMREAD_COLOR))
                    with patch.object(app, 'known_student_features', return_value=[]), patch.object(app, 'annotate_faces', side_effect=lambda frame, known: (frame, [])):
                        video_result = teacher.post('/api/recognize', files={'media': ('crowd.avi', video_path.read_bytes(), 'video/x-msvideo')})
                    self.assertEqual(video_result.status_code, 200, video_result.text)
                    result_url = video_result.json()['result_url']
                    encoded_result = teacher.get(result_url)
                    self.assertIn(b'avc1', encoded_result.content)
                    partial = teacher.get(result_url, headers={'Range': 'bytes=0-31'})
                    self.assertEqual(partial.status_code, 206)
                    self.assertEqual(partial.content, encoded_result.content[:32])
                    self.assertEqual(teacher.head(result_url).content, b'')
                    self.assertEqual(student.get(result_url, headers={'Range': 'bytes=0-31'}).status_code, 401)

                self.assertEqual(student.post('/api/attendance', json={'status': 'Present'}).status_code, 405)
                self.assertEqual(student.get(response.json()['result_url']).status_code, 401)
                self.assertEqual(student.post('/api/recognize', files={'media': ('x.jpg', photo.tobytes(), 'image/jpeg')}).status_code, 401)
                self.assertEqual(TestClient(app.app).get('/api/attendance').status_code, 401)
        finally:
            app.mongo_client.drop_database(db.name)


if __name__ == '__main__':
    unittest.main()
