"""Command-line interface for model inference."""
import argparse
import os
from pathlib import Path

from ortak.calisma_ortami import (
    ENGINES,
    RESTORERS,
    DemoError,
    block_inference_network,
    configure,
    emit,
    error_response,
    quiet_libraries,
)

ROOT = Path(__file__).resolve().parents[1]


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise DemoError('invalid_arguments')


SECTIONS = {'etiketleme': ENGINES, 'goruntu_iyilestirme': RESTORERS}


def parser(section=None):
    if section is not None and section not in SECTIONS:
        raise DemoError('invalid_arguments')
    program = f'{section}_calistir.py' if section else 'yerel_api_baslat.py'
    description = {
        'etiketleme': 'Object boxes and multilingual OCR; JSON output.',
        'goruntu_iyilestirme': 'Image deblurring; same-resolution PNG output.',
        None: 'Local JSON API for labeling and image restoration.',
    }[section]
    result = SafeParser(prog=program, description=description)
    commands = result.add_subparsers(dest='action', required=True)
    if section:
        download = commands.add_parser('download', help='Download models for this section.')
        download.add_argument('--engine', choices=(*SECTIONS[section], 'all'), default='all')
        download.add_argument('--allow-network', action='store_true')
        download.add_argument('--models', type=Path, default=ROOT/'models')
        check = commands.add_parser('check', help='Verify models for this section.')
        check.add_argument('--engine', choices=(*SECTIONS[section], 'all'), default='all')
        check.add_argument('--models', type=Path, default=ROOT/'models')
    if section == 'etiketleme':
        predict = commands.add_parser('predict', help='Run one image and print boxes or text as JSON.')
        predict.add_argument('--engine', choices=ENGINES, required=True)
        predict.add_argument('--image', type=Path, required=True)
        predict.add_argument('--models', type=Path, default=ROOT/'models')
        predict.add_argument('--prompt', default='')
        predict.add_argument('--target', action='append', default=[])
        predict.add_argument('--threshold', type=float, default=.5)
        predict.add_argument('--text-threshold', type=float, default=.25)
        predict.add_argument('--nms', type=float, default=.5)
    if section == 'goruntu_iyilestirme':
        restore = commands.add_parser('restore',help='Restore an image and write a same-resolution PNG.')
        restore.add_argument('--engine',choices=RESTORERS,required=True)
        restore.add_argument('--image',type=Path,required=True)
        restore.add_argument('--output',type=Path,required=True)
        restore.add_argument('--blend',type=float,default=1.)
        restore.add_argument('--models',type=Path,default=ROOT/'models')
    if section is None:
        serve = commands.add_parser('serve', prog='yerel_api_baslat.py', help='Run a local JSON API without a web interface.')
        serve.add_argument('--port', type=int, default=8090)
        serve.add_argument('--models', type=Path, default=ROOT/'models')
    return result


def main(argv=None, *, section=None):
    try:
        args = parser(section).parse_args(argv)
        configure(ROOT/'.cache')
        if args.action == 'download':
            from ortak.model_dosyalari import prepare
            engines = SECTIONS[section] if args.engine == 'all' else (args.engine,)
            with quiet_libraries():
                result = prepare(engines, args.models, args.allow_network)
        elif args.action == 'serve':
            if not 1024 <= args.port <= 65535:
                raise DemoError('invalid_arguments')
            from ortak.yerel_api_sunucusu import serve
            serve(args.models, args.port)
            return 0
        else:
            block_inference_network()
            with quiet_libraries():
                if args.action == 'check':
                    from ortak.model_dosyalari import verify
                    engines = SECTIONS[section] if args.engine == 'all' else (args.engine,)
                    for engine in engines:
                        verify(engine, args.models)
                    result = {'ok': True, 'verified': list(engines)}
                elif args.action == 'restore':
                    from goruntu_iyilestirme.iyilestirme_motorlari import (
                        RestorationRunner,
                    )

                    from ortak.gorsel_okuma import MAX_BYTES
                    if args.image.resolve()==args.output.resolve() or args.output.suffix.lower()!='.png':
                        raise DemoError('invalid_arguments')
                    if args.image.stat().st_size>MAX_BYTES:
                        raise DemoError('input_too_large')
                    data,size = RestorationRunner(args.engine,args.models).restore(args.image.read_bytes(),args.blend)
                    args.output.parent.mkdir(parents=True,exist_ok=True)
                    temporary = args.output.with_name(args.output.name+'.part')
                    try:
                        temporary.write_bytes(data)
                        os.replace(temporary,args.output)
                    finally:
                        temporary.unlink(missing_ok=True)
                    result = {'ok':True,'engine':args.engine,'image_size':{'width':size[0],'height':size[1]},'blend':args.blend}
                else:
                    from etiketleme.etiketleme_motorlari import EngineRunner

                    from ortak.gorsel_okuma import MAX_BYTES
                    if args.image.stat().st_size > MAX_BYTES:
                        raise DemoError('input_too_large')
                    result = EngineRunner(args.engine, args.models).predict(args.image.read_bytes(), prompt=args.prompt,
                        targets=args.target, threshold=args.threshold, text_threshold=args.text_threshold, nms=args.nms)
        emit(result)
        return 0
    except DemoError as exc:
        emit(error_response(exc.code))
    except (ImportError, ModuleNotFoundError):
        emit(error_response('dependencies_missing'))
    except KeyboardInterrupt:
        emit(error_response('operation_failed'))
    except Exception:
        emit(error_response('operation_failed'))
    return 1
