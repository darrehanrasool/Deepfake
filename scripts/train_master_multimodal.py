#!/usr/bin/env python3
"""DeepFake-AI: Master Multimodal Deepfake Training Pipeline (Script Alias).

Clean modular entrypoint delegating to university.main.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from university.main import main

if __name__ == '__main__':
    main()
