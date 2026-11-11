"""Start the shared localhost JSON API for both model groups."""
import sys

sys.dont_write_bytecode = True

if __name__ == '__main__':
    from ortak.komut_satiri import main
    raise SystemExit(main(['serve', *sys.argv[1:]]))
