"""Read a source/trace packet and print its derived evidence."""
import argparse
import json
from pathlib import Path

from . import analyze_packet

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="UTF-8 packet with source and recorded traceback")
    args = parser.parse_args()
    result = analyze_packet(args.packet.read_text(encoding="utf-8-sig"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
