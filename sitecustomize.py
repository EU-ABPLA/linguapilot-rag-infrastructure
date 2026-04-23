from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent / "src"
value = str(src_path)
if value not in sys.path:
    sys.path.insert(0, value)
