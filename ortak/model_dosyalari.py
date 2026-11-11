"""Public, fixed model files. Downloads are a separate opt-in operation."""
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from ortak.calisma_ortami import DemoError


def manifest():
    return json.loads(Path(__file__).with_name('model_surumleri.json').read_text('utf-8'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid_file(path, spec):
    return path.is_file() and path.stat().st_size == spec['bytes'] and sha256(path) == spec['sha256']


def components(engine):
    try:
        return manifest()['engines'][engine]['components']
    except KeyError:
        raise DemoError('invalid_request') from None


def verify(engine, root):
    folders = []
    for component in components(engine):
        directory = Path(root) / component['directory']
        for spec in component['files']:
            file = directory / spec['name']
            if not file.is_file():
                raise DemoError('model_missing')
            if not valid_file(file, spec):
                raise DemoError('model_integrity')
        folders.append(directory)
    return folders


def download_url(url):
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ''
    allowed = host == 'huggingface.co' or host == 'hf.co' or host.endswith('.hf.co') or host.endswith('.huggingface.co')
    if not allowed or parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise DemoError('download_failed')


class FixedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        download_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def prepare(engines, root, allow_network=False):
    if not allow_network:
        raise DemoError('network_disabled')
    # Use direct requests to the fixed public model sources.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), FixedRedirects())
    for engine in engines:
        if engine == 'concertormer-reds':
            from goruntu_iyilestirme.concertormer_agirlik_hazirlama import (
                prepare_concertormer,
            )
            prepare_concertormer(root)
            continue
        for component in components(engine):
            repository, revision = component['repository'], component['revision']
            if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+', repository) or not re.fullmatch(r'[a-f0-9]{40}', revision):
                raise DemoError('model_integrity')
            directory = Path(root) / component['directory']
            directory.mkdir(parents=True, exist_ok=True)
            for spec in component['files']:
                file = directory / spec['name']
                if file.exists() and valid_file(file, spec):
                    continue
                url = f'https://huggingface.co/{repository}/resolve/{revision}/{spec["name"]}'
                download_url(url)
                temporary = file.with_name(file.name + '.part')
                try:
                    request = urllib.request.Request(url, headers={'User-Agent': 'model-demo', 'Accept-Encoding': 'identity'})
                    with opener.open(request, timeout=60) as response, temporary.open('wb') as stream:
                        received = 0
                        while chunk := response.read(1024 * 1024):
                            received += len(chunk)
                            if received > spec['bytes']:
                                raise DemoError('download_failed')
                            stream.write(chunk)
                    if not valid_file(temporary, spec):
                        raise DemoError('model_integrity')
                    os.replace(temporary, file)
                finally:
                    temporary.unlink(missing_ok=True)
        verify(engine, root)
    return {'ok': True, 'prepared': list(engines)}
