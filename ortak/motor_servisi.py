"""Python interface shared by HTTP and programmatic experiments."""
import base64
import threading
from pathlib import Path

from ortak.calisma_ortami import (
    ENGINES,
    RESTORERS,
    DemoError,
    block_inference_network,
    configure,
    quiet_libraries,
)

_inference_lock = threading.Lock()


class InferenceService:
    def __init__(self, model_root, cache_root=None):
        self.model_root = Path(model_root)
        self.cache_root = Path(cache_root) if cache_root else self.model_root.parent/'.cache'
        self.runners = {}

    def predict(self, engine, image_bytes, **options):
        if engine not in ENGINES:
            raise DemoError('invalid_request')
        with _inference_lock, quiet_libraries():
            configure(self.cache_root)
            block_inference_network()
            from etiketleme.etiketleme_motorlari import EngineRunner
            if engine not in self.runners:
                self.runners[engine] = EngineRunner(engine, self.model_root)
            return self.runners[engine].predict(image_bytes, **options)

    def restore(self,engine,image_bytes,blend=1.):
        if engine not in RESTORERS:
            raise DemoError('invalid_request')
        with _inference_lock,quiet_libraries():
            configure(self.cache_root)
            block_inference_network()
            from goruntu_iyilestirme.iyilestirme_motorlari import RestorationRunner
            if engine not in self.runners:
                self.runners[engine] = RestorationRunner(engine,self.model_root)
            data,size = self.runners[engine].restore(image_bytes,blend)
            return {'ok':True,'engine':engine,'image_size':{'width':size[0],'height':size[1]},
                    'blend':blend,'image_base64':base64.b64encode(data).decode('ascii')}
