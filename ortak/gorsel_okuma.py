"""JPEG and PNG input decoding."""
import io

from ortak.calisma_ortami import DemoError

MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 20_000_000
MAX_EDGE = 8192


def image_from_bytes(data):
    if not isinstance(data, bytes) or not data:
        raise DemoError('invalid_image')
    if len(data) > MAX_BYTES:
        raise DemoError('input_too_large')
    from PIL import Image, ImageOps
    try:
        with Image.open(io.BytesIO(data), formats=('PNG', 'JPEG')) as source:
            if (source.width * source.height > MAX_PIXELS or max(source.size) > MAX_EDGE
                    or getattr(source, 'n_frames', 1) != 1):
                raise DemoError('input_too_large')
            pixels = ImageOps.exif_transpose(source).convert('RGB')
            clean = Image.new('RGB', pixels.size)
            clean.paste(pixels)
        return clean
    except DemoError:
        raise
    except Exception:
        raise DemoError('invalid_image') from None
