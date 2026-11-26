#!/usr/bin/env python3
"""Main Entrypoint Runner for Banking KYC Platform.

Author: chandan bauri <chandanbauri441@gmail.com>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from banking_kyc.cli import main

if __name__ == "__main__":
    main()
