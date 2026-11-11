"""Prepare the fixed Concertormer REDS weights for local inference."""
import hashlib
import json
import os
import shutil
import struct
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

from ortak.calisma_ortami import DemoError

CHECKPOINT = {
    'url': 'https://drive.usercontent.google.com/download?id=1NvbGroZm4vVgvWJtmgW-6Fwd97PiukiL&export=download&confirm=t',
    'archive_size': 1298912519,
    'entry': 'models/REDS/REDS.pth', 'offset': 135,
    'compressed': 187872662, 'bytes': 202554935, 'crc32': 345768577,
    'sha256': '4ccc84614b585167e6768ea6b3aeaf24299c8c0e7f4828a8e14384e8e8da3b31',
    'revision': '94269c556277805051644343e3a2cc76c4cd7aaf',
}


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def convert_checkpoint(source, target):
    source, target = Path(source), Path(target)
    if source.stat().st_size != CHECKPOINT['bytes'] or file_sha(source) != CHECKPOINT['sha256']:
        raise DemoError('model_integrity')
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    key = 'params' if 'params' in state else 'params_ema' if 'params_ema' in state else None
    if key:
        state = state[key]
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
            and v.dtype == torch.float32 and torch.isfinite(v).all() for k, v in state.items()):
        raise DemoError('model_integrity')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.convert')
    canonical = target.with_name(target.name+'.canonical')
    try:
        save_file({k:v.contiguous() for k,v in sorted(state.items())}, str(temporary),
            metadata={'format':'pt','source_revision':CHECKPOINT['revision'],'profile':'reds','checkpoint_key':key or 'root'})
        # Canonical header ordering makes output hashes reproducible.
        with temporary.open('rb') as src, canonical.open('wb') as dst:
            length = struct.unpack('<Q',src.read(8))[0]
            header = json.loads(src.read(length))
            encoded = json.dumps(header,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
            encoded += b' ' * (-len(encoded)%8)
            dst.write(struct.pack('<Q',len(encoded)))
            dst.write(encoded)
            shutil.copyfileobj(src,dst,1024*1024)
        os.replace(canonical,target)
    finally:
        temporary.unlink(missing_ok=True)
        canonical.unlink(missing_ok=True)


def check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in {'drive.google.com','drive.usercontent.google.com'} or parsed.username or parsed.password or parsed.port not in (None,443):
        raise DemoError('download_failed')


class DriveRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,message,headers,newurl):
        check_url(newurl)
        return super().redirect_request(request,fp,code,message,headers,newurl)


def download_checkpoint(target):
    target = Path(target)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),DriveRedirects())
    check_url(CHECKPOINT['url'])

    def open_range(start,end):
        request = urllib.request.Request(CHECKPOINT['url'],headers={
            'User-Agent':'model-demo','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        response = opener.open(request,timeout=60)
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{CHECKPOINT["archive_size"]}':
            response.close()
            raise DemoError('download_failed')
        return response

    def read_range(start,size):
        with open_range(start,start+size-1) as response:
            value = response.read(size+1)
        if len(value) != size:
            raise DemoError('download_failed')
        return value

    start = CHECKPOINT['offset']
    values = struct.unpack('<4s5H3L2H',read_range(start,30))
    if values[0] != b'PK\x03\x04' or values[3] != 8:
        raise DemoError('model_integrity')
    name_len,extra_len = values[-2:]
    entry = read_range(start+30,name_len+extra_len)
    if entry[:name_len].decode('utf-8') != CHECKPOINT['entry']:
        raise DemoError('model_integrity')
    data_start = start+30+name_len+extra_len
    unpacker = zlib.decompressobj(-15)
    count,received,crc = 0,0,0
    with open_range(data_start,data_start+CHECKPOINT['compressed']-1) as response,target.open('wb') as output:
        while chunk := response.read(256*1024):
            received += len(chunk)
            if received > CHECKPOINT['compressed']:
                raise DemoError('download_failed')
            value = unpacker.decompress(chunk,4*1024*1024)
            count += len(value)
            if unpacker.unconsumed_tail or count > CHECKPOINT['bytes']:
                raise DemoError('model_integrity')
            crc = zlib.crc32(value,crc)
            output.write(value)
        value = unpacker.flush()
        count += len(value)
        crc = zlib.crc32(value,crc)
        output.write(value)
    if (not unpacker.eof or unpacker.unused_data or count != CHECKPOINT['bytes']
            or received != CHECKPOINT['compressed'] or crc != CHECKPOINT['crc32'] or file_sha(target) != CHECKPOINT['sha256']):
        raise DemoError('model_integrity')


def prepare_concertormer(root):
    from ortak.model_dosyalari import components, valid_file, verify
    component = components('concertormer-reds')[0]
    directory = Path(root)/component['directory']
    directory.mkdir(parents=True,exist_ok=True)
    target = directory/'model.safetensors'
    if target.is_file() and valid_file(target,component['files'][0]):
        return
    checkpoint = directory/'.checkpoint.part'
    converted = directory/'.model.part'
    try:
        download_checkpoint(checkpoint)
        convert_checkpoint(checkpoint,converted)
        if not valid_file(converted,component['files'][0]):
            raise DemoError('model_integrity')
        os.replace(converted,target)
        verify('concertormer-reds',root)
    finally:
        checkpoint.unlink(missing_ok=True)
        converted.unlink(missing_ok=True)
