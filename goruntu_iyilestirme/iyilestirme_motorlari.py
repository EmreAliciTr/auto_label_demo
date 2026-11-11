"""Same-resolution NAFNet-REDS and Concertormer-REDS restoration."""
import io
import math
from contextlib import suppress

from ortak.calisma_ortami import RESTORERS, DemoError
from ortak.gorsel_okuma import image_from_bytes
from ortak.model_dosyalari import verify


def nafnet_tiles(original, session, input_name, blend):
    import numpy as np
    height, width = original.shape[:2]
    result = np.empty_like(original)
    for y in range(0,height,320):
        for x in range(0,width,320):
            y2,x2 = min(y+320,height),min(x+320,width)
            top,left = max(0,y-32),max(0,x-32)
            bottom,right = min(height,y2+32),min(width,x2+32)
            patch = original[top:bottom,left:right]
            ph,pw = patch.shape[:2]
            th,tw = max(384,((ph+15)//16)*16),max(384,((pw+15)//16)*16)
            patch = np.pad(patch,((0,th-ph),(0,tw-pw),(0,0)),mode='reflect' if min(ph,pw)>1 else 'edge')
            tensor = np.ascontiguousarray(patch.transpose(2,0,1)[None],dtype=np.float32)*np.float32(.00392)
            output = session.run(None,{input_name:tensor})[0]
            if output.shape != tensor.shape or not np.isfinite(output).all():
                raise DemoError('inference_failed')
            output = output[0].transpose(1,2,0)
            core = np.clip(output[y-top:y2-top,x-left:x2-left]*255.,0,255)
            if blend != 1:
                core = blend*core+(1-blend)*original[y:y2,x:x2]
            result[y:y2,x:x2] = core.astype(np.uint8)
    return result


def concertormer_tiles(original, model, blend):
    import numpy as np
    import torch
    height,width = original.shape[:2]
    core = max(width,height) if max(width,height)<=512 else 384
    result = np.empty_like(original)
    with torch.inference_mode():
        for y in range(0,height,core):
            for x in range(0,width,core):
                y2,x2 = min(y+core,height),min(x+core,width)
                top,left = max(0,y-64),max(0,x-64)
                bottom,right = min(height,y2+64),min(width,x2+64)
                patch = original[top:bottom,left:right]
                ph,pw = patch.shape[:2]
                th,tw = max(64,((ph+63)//64)*64),max(64,((pw+63)//64)*64)
                patch = np.pad(patch,((0,th-ph),(0,tw-pw),(0,0)),mode='reflect' if min(ph,pw)>1 else 'edge')
                tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2,0,1)[None],dtype=np.float32))/255.
                for module in model.modules():
                    if 'attn_mask' in module._buffers:
                        del module.attn_mask
                output = model(tensor)[-1]
                if output.shape != tensor.shape or not torch.isfinite(output).all():
                    raise DemoError('inference_failed')
                output = output[0].permute(1,2,0).numpy()
                value = np.clip(output[y-top:y2-top,x-left:x2-left]*255.,0,255)
                if blend != 1:
                    value = blend*value+(1-blend)*original[y:y2,x:x2]
                result[y:y2,x:x2] = np.rint(value).astype(np.uint8)
    return result


class RestorationRunner:
    def __init__(self,engine,model_root):
        if engine not in RESTORERS:
            raise DemoError('invalid_request')
        self.engine,self.model_root = engine,model_root
        self.model = None

    def load(self):
        if self.model is not None:
            return
        folder = verify(self.engine,self.model_root)[0]
        if self.engine == 'nafnet-reds':
            import onnxruntime as ort
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.log_severity_level = 3
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self.model = ort.InferenceSession(str(folder/'deblurring_nafnet_2025may.onnx'),sess_options=options,
                                              providers=['CPUExecutionProvider'])
            self.input_name = self.model.get_inputs()[0].name
        else:
            import torch
            from safetensors.torch import load_file

            from .concertormer_mimarisi.concertormer_model import ConcertormerLocal
            torch.set_num_threads(2)
            with suppress(RuntimeError):
                torch.set_num_interop_threads(1)
            model = ConcertormerLocal(width=48,ffn_expansion_factor=2,train_size=(1,3,256,256),
                grid_factor=2.,enc_blk_nums=[6,8,24],middle_blk_num=2,dec_blk_nums=[24,8,8],
                enc_heads=[1,2,4],middle_heads=8,dec_heads=[4,2,1])
            model.load_state_dict(load_file(str(folder/'model.safetensors'),device='cpu'),strict=True)
            self.model = model.eval()

    def restore(self,image_bytes,blend=1.):
        if not isinstance(blend,(float,int)) or isinstance(blend,bool) or not math.isfinite(blend) or not .1<=blend<=1:
            raise DemoError('invalid_request')
        image = image_from_bytes(image_bytes)
        limit = 4_000_000 if self.engine == 'nafnet-reds' else 1_000_000
        if image.width*image.height>limit:
            raise DemoError('input_too_large')
        self.load()
        import numpy as np
        from PIL import Image
        original = np.asarray(image,dtype=np.uint8)
        result = nafnet_tiles(original,self.model,self.input_name,blend) if self.engine == 'nafnet-reds' else concertormer_tiles(original,self.model,blend)
        if result.shape != original.shape or result.dtype != np.uint8:
            raise DemoError('inference_failed')
        stream = io.BytesIO()
        Image.fromarray(result).save(stream,format='PNG')
        return stream.getvalue(),image.size
