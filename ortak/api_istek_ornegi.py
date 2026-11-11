"""Local API request example."""
# ruff: noqa: E402 -- standalone tool adds only its own source root before imports.
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ortak.calisma_ortami import ENGINES, RESTORERS, DemoError, emit, error_response
from ortak.komut_satiri import SafeParser


def main():
    parser = SafeParser(prog='request-api')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--engine', choices=ENGINES+RESTORERS, required=True)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--prompt', default='')
    parser.add_argument('--blend',type=float,default=1.)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.image.stat().st_size > 16*1024*1024:
        raise DemoError('invalid_arguments')
    restoration = args.engine in RESTORERS
    if restoration and (not args.output or args.output.suffix.lower()!='.png' or args.output.resolve()==args.image.resolve()):
        raise DemoError('invalid_arguments')
    # Use PNG encoding for the image payload.
    import io

    from ortak.gorsel_okuma import image_from_bytes
    image = image_from_bytes(args.image.read_bytes())
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    body = {'engine':args.engine, 'image_base64':base64.b64encode(buffer.getvalue()).decode('ascii')}
    body.update({'blend':args.blend} if restoration else {'prompt':args.prompt})
    endpoint = 'restore' if restoration else 'predict'
    request = urllib.request.Request(f'http://127.0.0.1:{args.port}/{endpoint}', data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type':'application/json','User-Agent':'model-demo'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=600) as response:
            result = json.load(response)
        if restoration:
            result_image = base64.b64decode(result.pop('image_base64'),validate=True)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_bytes(result_image)
        emit(result)
    except urllib.error.HTTPError as error:
        emit(json.load(error))
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except DemoError as error:
        emit(error_response(error.code))
        raise SystemExit(1) from None
    except Exception:
        emit(error_response('operation_failed'))
        raise SystemExit(1) from None
