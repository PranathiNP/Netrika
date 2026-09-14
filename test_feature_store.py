import unittest
from unittest.mock import Mock
from uuid import uuid4

import cv2
import numpy as np
from gridfs import GridFS
import app
from feature_store import load_student_features


class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = app.mongo_client['netrika_test_' + uuid4().hex]
        self.addCleanup(lambda: app.mongo_client.drop_database(self.db.name))
        self.storage = GridFS(self.db, collection='face_frames')
        self.user = self.db.users.insert_one({'full_name': 'Alex', 'role': 'student'}).inserted_id
        _, encoded = cv2.imencode('.jpg', np.zeros((64, 64, 3), dtype=np.uint8))
        self.content = encoded.tobytes()
        self.extract = Mock(return_value=np.ones(128, dtype=np.float32) / np.sqrt(128))

    def put(self, pending=False):
        return self.storage.put(self.content, metadata={'user_id': str(self.user), 'pending': pending, 'face_box': [0, 0, 64, 64]})

    def load(self, version='v1'):
        return load_student_features(self.db, self.storage, self.extract, version)

    def test_reuses_features_and_invalidates_changed_model(self):
        self.put()
        self.assertEqual(len(self.load()), 1)
        self.load()
        self.assertEqual(self.extract.call_count, 1)
        self.load('v2')
        self.assertEqual(self.extract.call_count, 2)

    def test_deleted_enrollment_and_users_never_remain_cached(self):
        file_id = self.put()
        self.put(pending=True)
        self.load()
        self.assertEqual(self.extract.call_count, 1)
        self.storage.delete(file_id)
        self.assertEqual(self.load(), [])
        self.put()
        self.db.users.delete_one({'_id': self.user})
        self.assertEqual(self.load(), [])

    def test_corrupt_images_skipped_and_bad_cache_recomputed(self):
        self.storage.put(b'invalid', metadata={'user_id': str(self.user), 'face_box': [0, 0, 64, 64]})
        self.assertEqual(self.load(), [])
        file_id = self.put()
        self.db['face_frames.files'].update_one({'_id': file_id}, {'$set': {'metadata.feature': [float('nan')] * 128, 'metadata.feature_version': 'v1'}})
        self.assertEqual(len(self.load()), 1)
        self.assertEqual(self.extract.call_count, 1)


if __name__ == '__main__':
    unittest.main()
