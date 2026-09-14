from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from media_delivery import browser_video, stored_media_response


class MediaDeliveryTests(unittest.TestCase):
    def setUp(self):
        api = FastAPI()
        @api.api_route('/video', methods=['GET', 'HEAD'])
        def result(request: Request):
            stored = BytesIO(b'0123456789')
            stored.length = 10
            stored.filename = 'video.mp4'
            stored.content_type = 'video/mp4'
            return stored_media_response(stored, request)
        self.client = TestClient(api)

    def test_complete_and_head(self):
        response = self.client.get('/video')
        self.assertEqual(response.content, b'0123456789')
        self.assertEqual(response.headers['content-length'], '10')
        self.assertEqual(response.headers['accept-ranges'], 'bytes')
        self.assertEqual(self.client.head('/video').content, b'')
        self.assertEqual(self.client.head('/video').headers['content-length'], '10')

    def test_browser_ranges(self):
        for header, expected in [('bytes=0-1', b'01'), ('bytes=7-', b'789'), ('bytes=-3', b'789'), ('bytes=8-99', b'89')]:
            response = self.client.get('/video', headers={'Range': header})
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.content, expected)
            self.assertIn('content-range', response.headers)
        for header in ['bytes=99-', 'bytes=-0', 'bytes=5-2', 'bytes=-']:
            response = self.client.get('/video', headers={'Range': header})
            self.assertEqual(response.status_code, 416)
            self.assertEqual(response.headers['content-range'], 'bytes */10')

    def test_h264_faststart_output_decodes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.mp4', Path(directory) / 'output.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(np.full((64, 64, 3), index * 20, dtype=np.uint8))
            writer.release()
            browser_video(source, output)
            payload = output.read_bytes()
            self.assertIn(b'avc1', payload)
            self.assertLess(payload.index(b'moov'), payload.index(b'mdat'))
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.read()[0])
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            finally:
                capture.release()


if __name__ == '__main__':
    unittest.main()
