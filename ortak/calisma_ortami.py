"""Runtime settings and common error handling."""
import contextlib
import json
import logging
import os
import socket
import sys
import warnings
from pathlib import Path

ENGINES = ('grounding-dino-tiny', 'ppocr-v6-medium')
RESTORERS = ('nafnet-reds', 'concertormer-reds')
ERRORS = {
    'invalid_arguments': 'Check the command and its options.',
    'invalid_request': 'Check the input values.',
    'invalid_image': 'Use a valid single-frame JPEG or PNG within the limits.',
    'input_too_large': 'The input exceeds the configured limit.',
    'model_missing': 'Prepare the selected model before inference.',
    'model_integrity': 'Model verification failed; prepare the model again.',
    'dependencies_missing': 'Install the listed requirements in an isolated environment.',
    'inference_failed': 'Inference failed; check input, limits and model readiness.',
    'operation_failed': 'The operation could not be completed.',
    'network_disabled': 'Model download requires the explicit --allow-network option.',
    'download_failed': 'Download failed; no unverified file was accepted.',
}


class DemoError(Exception):
    def __init__(self, code):
        self.code = code if code in ERRORS else 'operation_failed'
        super().__init__(self.code)


def error_response(code):
    code = code if code in ERRORS else 'operation_failed'
    return {'ok': False, 'error': {'code': code, 'message': ERRORS[code]}}


def emit(value):
    print(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')))


def configure(cache):
    """Configure local model execution."""
    cache = Path(cache)
    settings = {
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
        'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'HF_HUB_DISABLE_XET': '1',
        'DO_NOT_TRACK': '1', 'OTEL_SDK_DISABLED': 'true',
        'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
        'HF_HOME': str(cache / 'hf'), 'TORCH_HOME': str(cache / 'torch'),
    }
    os.environ.update(settings)
    sys.dont_write_bytecode = True
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings('ignore')


def block_inference_network():
    """Keep inference local."""
    if getattr(socket, '_demo_outbound_blocked', False):
        return
    for name in ('connect', 'connect_ex', 'sendto', 'sendmsg'):
        if not hasattr(socket.socket, name):
            continue
        original = getattr(socket.socket, name)

        def guarded(self, *args, _original=original, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise PermissionError('Outbound inference networking is disabled')
            return _original(self, *args, **kwargs)

        setattr(socket.socket, name, guarded)

    def denied(*args, **kwargs):
        raise PermissionError('Outbound inference networking is disabled')

    for name in ('getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr', 'create_connection'):
        setattr(socket, name, denied)
    socket._demo_outbound_blocked = True


@contextlib.contextmanager
def quiet_libraries():
    """Control library output during model execution."""
    with open(os.devnull, 'w', encoding='utf-8') as sink:
        saved = []
        try:
            for descriptor in (1, 2):
                saved.append((descriptor, os.dup(descriptor)))
                os.dup2(sink.fileno(), descriptor)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            for descriptor, duplicate in saved:
                os.dup2(duplicate, descriptor)
                os.close(duplicate)
