"""Grounding DINO Tiny and PP-OCRv6 medium inference."""
import inspect
import math
import unicodedata

from ortak.calisma_ortami import ENGINES, DemoError
from ortak.gorsel_okuma import image_from_bytes
from ortak.model_dosyalari import verify


def normalize_text(text):
    value = unicodedata.normalize('NFKC', text).translate(str.maketrans({'I': 'ı', 'İ': 'i'}))
    return ' '.join(value.casefold().split())


def iou(a, b):
    intersection = max(0, min(a['x2'], b['x2']) - max(a['x1'], b['x1'])) * max(0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
    area_a = (a['x2']-a['x1'])*(a['y2']-a['y1'])
    area_b = (b['x2']-b['x1'])*(b['y2']-b['y1'])
    return intersection / (area_a+area_b-intersection) if intersection else 0.0


def filter_boxes(boxes, scores, width, height, threshold, nms):
    accepted = []
    for coordinates, score in zip(boxes, scores, strict=True):
        if len(coordinates) != 4 or not all(math.isfinite(float(v)) for v in [*coordinates, score]) or not threshold <= score <= 1:
            continue
        x1, y1, x2, y2 = map(float, coordinates)
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        if x2-x1 >= 1 and y2-y1 >= 1:
            accepted.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'score': float(score)})
    kept = []
    for box in sorted(accepted, key=lambda b: b['score'], reverse=True):
        if all(iou(box, old) <= nms for old in kept):
            kept.append(box)
    return kept


def ordered_polygon(points, width, height):
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise ValueError('Invalid polygon')
    points = [(float(x), float(y)) for x, y in points]
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError('Invalid polygon')
    points = [(max(0., min(width, x)), max(0., min(height, y))) for x, y in points]
    cx, cy = sum(x for x, _ in points)/4, sum(y for _, y in points)/4
    points.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
    start = min(range(4), key=lambda i: sum(points[i]))
    return points[start:] + points[:start]


def rectify(image, polygon):
    import cv2
    import numpy as np
    width = round(max(math.dist(polygon[0], polygon[1]), math.dist(polygon[3], polygon[2])))
    height = round(max(math.dist(polygon[0], polygon[3]), math.dist(polygon[1], polygon[2])))
    if min(width, height) < 2 or max(width, height) > 8192:
        raise ValueError('Degenerate crop')
    destination = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    transform = cv2.getPerspectiveTransform(np.float32(polygon), destination)
    crop = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return (np.rot90(crop) if height/width >= 1.5 else crop).copy()


class EngineRunner:
    """Reusable in-memory model returning image-space regions."""
    def __init__(self, engine, model_root):
        if engine not in ENGINES:
            raise DemoError('invalid_request')
        self.engine, self.model_root = engine, model_root
        self.model = self.processor = None

    def load(self):
        if self.model is not None:
            return
        folders = verify(self.engine, self.model_root)
        import torch
        torch.set_num_threads(2)
        if self.engine == 'grounding-dino-tiny':
            from transformers import (
                AutoConfig,
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
            folder = str(folders[0])
            config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
            config.disable_custom_kernels = True
            processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False, use_fast=False)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(folder, config=config,
                local_files_only=True, trust_remote_code=False, use_safetensors=True).to('cpu').eval()
            self.processor, self.model = processor, model
        else:
            from transformers import (
                AutoImageProcessor,
                AutoModelForObjectDetection,
                AutoModelForTextRecognition,
            )
            kwargs = {'local_files_only': True, 'trust_remote_code': False, 'use_safetensors': True,
                      'output_loading_info': True, 'dtype': torch.float32}
            detector, det_info = AutoModelForObjectDetection.from_pretrained(str(folders[0]), **kwargs)
            recognizer, rec_info = AutoModelForTextRecognition.from_pretrained(str(folders[1]), **kwargs)
            for info in (det_info, rec_info):
                if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                    raise DemoError('model_integrity')
            self.det_processor = AutoImageProcessor.from_pretrained(str(folders[0]), local_files_only=True, trust_remote_code=False)
            self.rec_processor = AutoImageProcessor.from_pretrained(str(folders[1]), local_files_only=True, trust_remote_code=False)
            self.model = detector.to('cpu').eval()
            self.recognizer = recognizer.to('cpu').eval()

    def predict(self, image_bytes, *, prompt='', targets=(), threshold=.5, text_threshold=.25, nms=.5):
        if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1
                    for v in (threshold, text_threshold, nms))
                or not isinstance(prompt, str) or len(prompt) > 250
                or not isinstance(targets, (list, tuple)) or len(targets) > 20
                or any(not isinstance(t, str) or not t.strip() or len(t) > 100 for t in targets)):
            raise DemoError('invalid_request')
        if self.engine == 'grounding-dino-tiny' and not prompt.strip():
            raise DemoError('invalid_request')
        image = image_from_bytes(image_bytes)
        self.load()
        import torch
        with torch.inference_mode():
            regions = self._dino(image, prompt, threshold, text_threshold, nms) if self.engine == 'grounding-dino-tiny' else self._ocr(image, targets, threshold)
        return {'ok': True, 'engine': self.engine, 'image_size': {'width': image.width, 'height': image.height}, 'regions': regions}

    def _dino(self, image, prompt, threshold, text_threshold, nms):
        text = prompt.strip().lower()
        if not text.endswith('.'):
            text += '.'
        inputs = self.processor(images=image, text=text, return_tensors='pt').to('cpu')
        outputs = self.model(**inputs)
        postprocess = self.processor.post_process_grounded_object_detection
        key = 'threshold' if 'threshold' in inspect.signature(postprocess).parameters else 'box_threshold'
        result = postprocess(outputs, input_ids=inputs.input_ids, target_sizes=[(image.height, image.width)],
                             text_threshold=text_threshold, **{key: threshold})[0]
        return filter_boxes(result['boxes'].detach().cpu().tolist(), result['scores'].detach().cpu().tolist(),
                            image.width, image.height, threshold, nms)

    def _ocr(self, image, targets, threshold):
        import numpy as np
        pixels = np.asarray(image)
        inputs = self.det_processor(images=image, return_tensors='pt', limit_type='min', limit_side_len=736, max_side_limit=1280)
        detection = self.model(pixel_values=inputs['pixel_values'])
        result = self.det_processor.post_process_object_detection(detection, target_sizes=inputs['target_sizes'],
            threshold=.3, box_threshold=.5, max_candidates=1000)[0]
        boxes, scores = result['boxes'].tolist(), result['scores'].tolist()
        if len(boxes) > 1000 or len(boxes) != len(scores):
            raise DemoError('inference_failed')
        wanted = {normalize_text(t) for t in targets}
        regions = []
        for points, detector_score in zip(boxes, scores, strict=True):
            if not math.isfinite(detector_score) or not 0 <= detector_score <= 1:
                continue
            try:
                polygon = ordered_polygon(points, image.width, image.height)
                crop = rectify(pixels, polygon)
            except ValueError:
                continue
            decoded = self.rec_processor.post_process_text_recognition(
                self.recognizer(**self.rec_processor(images=crop, return_tensors='pt')))[0]
            text, score = decoded['text'], float(decoded['score'])
            if (not text.strip() or not math.isfinite(score) or not threshold <= score <= 1
                    or (wanted and normalize_text(text) not in wanted)):
                continue
            regions.append({'text': text, 'score': score, 'detector_score': float(detector_score),
                            'polygon': [list(p) for p in polygon]})
        return sorted(regions, key=lambda r: r['score'], reverse=True)
