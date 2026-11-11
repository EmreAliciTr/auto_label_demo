"""Run NAFNet-REDS or Concertormer-REDS and save a restored PNG."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from ortak.komut_satiri import main
    raise SystemExit(main(section='goruntu_iyilestirme'))
