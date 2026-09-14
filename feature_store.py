"""Persist derived enrollment features without caching student ownership."""
import cv2
import numpy as np
from gridfs.errors import NoFile


def load_student_features(database, storage, extract, version):
    users = {str(user['_id']): user['full_name'] for user in
             database['users'].find({'role': 'student'}, {'full_name': 1})}
    if not users:
        return []
    grouped = {}
    files = database['face_frames.files']
    query = {'metadata.user_id': {'$in': list(users)}, 'metadata.pending': {'$ne': True}}
    for stored in files.find(query):
        metadata = stored.get('metadata', {})
        feature = None
        cached = metadata.get('feature_version') == version and 'feature' in metadata
        if cached and metadata['feature'] is not None:
            try:
                feature = np.asarray(metadata['feature'], dtype=np.float32)
                cached = feature.shape == (128,) and np.isfinite(feature).all() and np.linalg.norm(feature) > 0
            except (TypeError, ValueError):
                cached = False
        if not cached:
            try:
                with storage.get(stored['_id']) as image_file:
                    content = image_file.read()
            except NoFile:
                continue  # An enrollment update removed the file during this read.
            image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR) if content else None
            box = metadata.get('face_box')
            feature = extract(image, box) if image is not None and box is not None and len(box) == 4 else None
            if feature is not None:
                feature = np.asarray(feature, dtype=np.float32)
                if feature.shape != (128,) or not np.isfinite(feature).all() or np.linalg.norm(feature) <= 0:
                    feature = None
            files.update_one({'_id': stored['_id']}, {'$set': {
                'metadata.feature': feature.tolist() if feature is not None else None,
                'metadata.feature_version': version,
            }})
        if feature is not None:
            grouped.setdefault(metadata['user_id'], []).append(feature / np.linalg.norm(feature))
    known = []
    for user_id, features in grouped.items():
        average = np.mean(features, axis=0)
        norm = np.linalg.norm(average)
        if np.isfinite(norm) and norm > 0:
            known.append((user_id, users[user_id], average / norm))
    return known
