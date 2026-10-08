import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cams import data, release

if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(1)
    print(release.fill(sys.argv[1], data.load(sys.argv[2]), sys.argv[3]))
