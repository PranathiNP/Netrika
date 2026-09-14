import unittest
from unittest.mock import Mock, patch
import numpy as np
import app
from face_matching import match_faces


class FaceMatchingTests(unittest.TestCase):
    def test_one_identity_only_goes_to_clear_winner(self):
        known = [('a', 'Alex', np.array([1., 0.]))]
        self.assertEqual(match_faces([np.array([.9, .43589]), np.array([.6, .8])], known), [('a', 'Alex'), None])

    def test_competing_faces_with_similar_scores_both_unknown(self):
        known = [('a', 'Alex', np.array([1., 0.]))]
        self.assertEqual(match_faces([np.array([.8, .6]), np.array([.79, .61])], known), [None, None])

    def test_ambiguous_students_are_unknown_without_fallback(self):
        known = [('a', 'Alex', np.array([1., 0.])), ('b', 'Sam', np.array([.99, .141]))]
        self.assertEqual(match_faces([np.array([1., 0.])], known), [None])

    def test_threshold_and_unusable_features(self):
        known = [('a', 'Alex', np.array([1., 0.]))]
        self.assertEqual(match_faces([None, np.array([.49, .87]), np.array([np.nan, 0.])], known), [None, None, None])

    def test_same_names_do_not_merge_distinct_ids(self):
        known = [('a', 'Alex', np.array([1., 0.])), ('b', 'Alex', np.array([0., 1.]))]
        self.assertEqual(match_faces([np.array([1., 0.]), np.array([0., 1.])], known), [('a', 'Alex'), ('b', 'Alex')])

    def test_alignment_required_and_applied(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        detector, recognizer = Mock(), Mock()
        # Crop begins at (20,20); detected face corresponds to original box.
        detector.detect.return_value = (1, np.array([[20, 20, 80, 80, 40, 45, 75, 45, 60, 65, 45, 80, 75, 80, .99]], dtype=np.float32))
        recognizer.alignCrop.return_value = np.ones((112, 112, 3), dtype=np.uint8)
        recognizer.feature.return_value = np.array([[3., 4.]])
        with patch.object(app, 'get_landmark_model', return_value=detector), patch.object(app, 'get_recognition_model', return_value=recognizer):
            np.testing.assert_allclose(app.face_feature(image, [40, 40, 120, 120]), [.6, .8])
            recognizer.alignCrop.assert_called_once()
            detector.detect.return_value = (0, None)
            self.assertIsNone(app.face_feature(image, [40, 40, 120, 120]))
            self.assertEqual(recognizer.feature.call_count, 1)

    def test_small_faces_are_rejected(self):
        self.assertIsNone(app.face_feature(np.zeros((80, 80, 3)), [10, 10, 25, 25]))


if __name__ == '__main__':
    unittest.main()
