"""Restoration geometry, blending, limits and HTTP contracts."""
import base64
import io
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from goruntu_iyilestirme.concertormer_agirlik_hazirlama import download_checkpoint
from goruntu_iyilestirme.iyilestirme_motorlari import (
    RestorationRunner,
    concertormer_tiles,
    nafnet_tiles,
)
from ortak.calisma_ortami import DemoError
from ortak.yerel_api_sunucusu import create_app
from PIL import Image


def image_bytes():
    stream = io.BytesIO()
    Image.new('RGB',(40,24),'white').save(stream,'PNG')
    return stream.getvalue()


class RestorationTests(unittest.TestCase):
    def test_examples_have_the_same_pixel_dimensions(self):
        root = Path(__file__).resolve().parents[1]/'goruntu_iyilestirme/ornek_gorseller'
        for name in ('01_orijinal.png','02_concertormer_reds_yuzde100.png','03_nafnet_reds_yuzde100.png'):
            with Image.open(root/name) as image:
                self.assertEqual(image.size,(962,555))
                self.assertEqual(image.mode,'RGB')
                self.assertEqual(image.info,{})

    def test_checkpoint_download_refuses_full_archive_response(self):
        response = Mock(status=200,headers={})
        opener = Mock()
        opener.open.return_value = response
        with patch('urllib.request.build_opener',return_value=opener), self.assertRaisesRegex(DemoError,'^download_failed$'):
            download_checkpoint(Path('unused.pth'))
        response.close.assert_called_once()
        response.read.assert_not_called()

    def test_nafnet_tile_halos_and_padding_preserve_dimensions(self):
        image = np.full((555,962,3),64,dtype=np.uint8)
        session = Mock()
        shapes = []
        def process(_,inputs):
            tensor = inputs['pixels']
            shapes.append(tensor.shape)
            return [np.full(tensor.shape,.5,dtype=np.float32)]
        session.run.side_effect = process
        restored = nafnet_tiles(image,session,'pixels',1.)
        self.assertEqual(restored.shape,image.shape)
        self.assertEqual(len(shapes),8)
        self.assertTrue(all(h>=384 and w>=384 and h%16==w%16==0 for _,_,h,w in shapes))
        self.assertTrue(np.all(restored==127))

    def test_nafnet_blend_is_not_an_output_scale(self):
        image = np.full((25,37,3),100,dtype=np.uint8)
        session = Mock()
        session.run.side_effect = lambda _,inputs:[np.zeros_like(inputs['pixels'])]
        result = nafnet_tiles(image,session,'pixels',.5)
        self.assertEqual(result.shape,(25,37,3))
        self.assertTrue(np.all(result==50))

    def test_concertormer_tiles_remove_only_padding(self):
        import torch
        image = np.full((555,962,3),80,dtype=np.uint8)
        class Identity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes = []
                self.register_buffer('attn_mask',torch.zeros(1))
            def forward(self,value):
                self.shapes.append(tuple(value.shape))
                return [value]
        model = Identity()
        result = concertormer_tiles(image,model,1.)
        self.assertEqual(len(model.shapes),6)
        self.assertTrue(all(h%64==w%64==0 for _,_,h,w in model.shapes))
        self.assertTrue(np.array_equal(result,image))
        self.assertNotIn('attn_mask',model._buffers)

    def test_limits_are_checked_before_model_loading(self):
        runner = RestorationRunner('concertormer-reds',Path('models'))
        stream = io.BytesIO()
        Image.new('RGB',(1001,1000)).save(stream,'PNG')
        with patch.object(runner,'load') as load:
            with self.assertRaisesRegex(DemoError,'^input_too_large$'):
                runner.restore(stream.getvalue())
            for blend in (float('nan'),True,0,1.1):
                with self.assertRaisesRegex(DemoError,'^invalid_request$'):
                    runner.restore(image_bytes(),blend)
            load.assert_not_called()

    def test_png_result_keeps_pixel_size(self):
        runner = RestorationRunner('nafnet-reds',Path('models'))
        with patch.object(runner,'load'), patch('goruntu_iyilestirme.iyilestirme_motorlari.nafnet_tiles',side_effect=lambda pixels,*args:pixels):
            runner.model,runner.input_name = object(),'pixels'
            result,size = runner.restore(image_bytes())
        with Image.open(io.BytesIO(result)) as image:
            self.assertEqual(size,(40,24))
            self.assertEqual(image.size,size)
            self.assertEqual(image.info,{})


class HttpRestorationTests(unittest.TestCase):
    def test_restore_endpoint_and_request_validation(self):
        service = Mock()
        service.restore.return_value = {'ok':True,'engine':'nafnet-reds','image_size':{'width':40,'height':24},
                                        'blend':1.,'image_base64':base64.b64encode(image_bytes()).decode('ascii')}
        payload = {'engine':'nafnet-reds','image_base64':base64.b64encode(image_bytes()).decode('ascii')}
        with TestClient(create_app(Path('models'),service),base_url='http://127.0.0.1:8090') as client:
            response = client.post('/restore',json=payload)
            self.assertEqual(response.status_code,200)
            self.assertEqual(base64.b64decode(response.json()['image_base64']),image_bytes())
            service.restore.assert_called_once_with('nafnet-reds',image_bytes(),blend=1.)
            self.assertEqual(client.post('/restore',json={**payload,'blend':1.5}).status_code,422)
            self.assertEqual(client.post('/restore',json={**payload,'image_base64':'invalid!'}).status_code,400)


if __name__=='__main__':
    unittest.main()
