"""Unit, CLI and HTTP tests."""
import base64
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from etiketleme.etiketleme_motorlari import (
    filter_boxes,
    normalize_text,
    ordered_polygon,
)
from fastapi.testclient import TestClient
from ortak import model_dosyalari as artifacts
from ortak.calisma_ortami import DemoError
from ortak.gorsel_okuma import image_from_bytes
from ortak.yerel_api_sunucusu import create_app
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def synthetic_image():
    stream = io.BytesIO()
    Image.new('RGB', (80, 40), 'white').save(stream, format='PNG')
    return stream.getvalue()


class PixelTests(unittest.TestCase):
    def test_image_orientation_and_conversion(self):
        stream = io.BytesIO()
        source = Image.new('RGB', (80, 40), 'white')
        exif = Image.Exif()
        exif[274] = 6
        exif[315] = 'SAMPLE_VALUE'
        source.save(stream, format='JPEG', exif=exif)
        clean = image_from_bytes(stream.getvalue())
        self.assertEqual(clean.size, (40, 80))
        self.assertEqual(clean.info, {})
        self.assertFalse(clean.getexif())

    def test_invalid_image_format(self):
        with self.assertRaises(DemoError) as caught:
            image_from_bytes(b'SAMPLE_VALUE')
        self.assertEqual(str(caught.exception), 'invalid_image')

    def test_turkish_exact_text_normalization(self):
        self.assertEqual(normalize_text('  IŞIK  '), normalize_text('ışık'))
        self.assertEqual(normalize_text('ÜRÜN İÇİN'), normalize_text('ürün için'))
        self.assertNotEqual(normalize_text('URUN'), normalize_text('ÜRÜN'))

    def test_dino_filters_invalid_and_duplicate_boxes(self):
        result = filter_boxes([[-5, -5, 80, 40], [0, 0, 80, 40], [1, 1, 1, 5], [float('nan'), 1, 9, 9]],
                              [.95, .8, .9, .9], 80, 40, .5, .5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {'x1':0., 'y1':0., 'x2':80., 'y2':40., 'score':.95})

    def test_ocr_keeps_rotated_corners(self):
        points = [[70, 35], [10, 5], [65, 0], [15, 40]]
        ordered = ordered_polygon(points, 80, 40)
        self.assertEqual({tuple(p) for p in ordered}, {tuple(p) for p in points})
        self.assertEqual(ordered[0], (10., 5.))


class ArtifactTests(unittest.TestCase):
    def test_fixed_model_profiles(self):
        self.assertEqual(set(artifacts.manifest()['engines']), {'grounding-dino-tiny', 'ppocr-v6-medium','nafnet-reds','concertormer-reds'})
        for engine in artifacts.manifest()['engines']:
            for component in artifacts.components(engine):
                self.assertRegex(component['revision'], r'^[a-f0-9]{40}$')
                for file in component['files']:
                    self.assertRegex(file['sha256'], r'^[a-f0-9]{64}$')

    def test_prepare_requires_explicit_network_permission(self):
        with patch('urllib.request.build_opener') as network:
            with self.assertRaisesRegex(DemoError, '^network_disabled$'):
                artifacts.prepare(('grounding-dino-tiny',), Path('models'))
            network.assert_not_called()

    def test_integrity_rejects_modified_file(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'model').mkdir()
            (root/'model/weights').write_bytes(b'wrong')
            component = {'directory':'model','files':[{'name':'weights','bytes':5,'sha256':hashlib.sha256(b'right').hexdigest()}]}
            with patch.object(artifacts, 'components', return_value=[component]), self.assertRaisesRegex(DemoError, '^model_integrity$'):
                artifacts.verify('grounding-dino-tiny', root)



class FakeService:
    def __init__(self):
        self.received = []

    def predict(self, engine, data, **options):
        self.received.append((engine, data, options))
        return {'ok':True,'engine':engine,'image_size':{'width':80,'height':40},'regions':[]}


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.client = TestClient(create_app(Path('models'), self.service), base_url='http://127.0.0.1:8090', raise_server_exceptions=False)
        self.payload = {'engine':'ppocr-v6-medium','image_base64':base64.b64encode(synthetic_image()).decode('ascii')}

    def tearDown(self):
        self.client.close()

    def test_json_endpoint_and_minimal_health(self):
        response = self.client.post('/predict', json=self.payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(self.service.received[0][1], synthetic_image())
        self.assertEqual(set(self.client.get('/health').json()), {'ok','engines'})
        self.assertEqual(self.client.get('/docs').status_code, 404)

    def test_request_validation(self):
        response = self.client.post('/predict', json={**self.payload,'engine':'SAMPLE_VALUE','extra':'SAMPLE_VALUE'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('SAMPLE_VALUE', response.text)
        self.assertFalse(self.service.received)

    def test_malformed_base64_is_fixed_error(self):
        response = self.client.post('/predict', json={**self.payload,'image_base64':'not base64!'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['error']['code'], 'invalid_image')

    def test_cross_origin_request_is_rejected(self):
        response = self.client.post('/predict', json=self.payload, headers={'Origin':'https://example.invalid'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.service.received)

    def test_body_limit_and_processing_error(self):
        with patch('ortak.yerel_api_sunucusu.MAX_REQUEST', 10):
            self.assertEqual(self.client.post('/predict', json=self.payload).status_code, 413)
        with patch.object(self.service,'predict',side_effect=RuntimeError('SAMPLE_VALUE')):
            response = self.client.post('/predict', json=self.payload)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn('SAMPLE_VALUE',response.text)


class CliTests(unittest.TestCase):
    def call(self, *args):
        section = 'goruntu_iyilestirme' if args[0] == 'restore' else 'etiketleme'
        entry = ROOT/section/f'{section}_calistir.py'
        return subprocess.run([sys.executable,'-B',str(entry),*args],cwd=ROOT,
                              capture_output=True,text=True,timeout=30)

    def test_sections_reject_the_other_model_group(self):
        for section, engine in (('etiketleme','nafnet-reds'), ('goruntu_iyilestirme','ppocr-v6-medium')):
            with self.subTest(section=section):
                entry = ROOT/section/f'{section}_calistir.py'
                result = subprocess.run([sys.executable,'-B',str(entry),'check','--engine',engine],cwd=ROOT,
                                        capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(json.loads(result.stdout)['error']['code'],'invalid_arguments')
                self.assertEqual(result.stderr,'')

    def test_invalid_arguments(self):
        result = self.call('predict','--engine','SAMPLE_VALUE','--image','not-present.png')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(json.loads(result.stdout)['error']['code'],'invalid_arguments')
        self.assertNotIn('SAMPLE_VALUE',result.stdout+result.stderr)
        self.assertEqual(result.stderr,'')

    def test_missing_models(self):
        result = self.call('check','--models','SAMPLE_VALUE')
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(json.loads(result.stdout)['error']['code'],'model_missing')
        self.assertNotIn('SAMPLE_VALUE',result.stdout+result.stderr)
        self.assertEqual(result.stderr,'')

    def test_restoration_cannot_overwrite_its_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.png'
            original = synthetic_image()
            path.write_bytes(original)
            result = self.call('restore','--engine','nafnet-reds','--image',str(path),'--output',str(path))
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(json.loads(result.stdout)['error']['code'],'invalid_arguments')
            self.assertEqual(path.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
