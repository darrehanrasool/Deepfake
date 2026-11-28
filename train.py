#!/usr/bin/env python3
"""DeepFake-AI: Master Multimodal Deepfake Training Pipeline.

Professional, modular entrypoint delegating to university.main.

Institutional Authors:
- Dar Rehan Rasool (M.Tech Student, Centre for Artificial Intelligence, IUST)
- Dr. Riyaz Ahmed (M.Tech Student, Centre for Artificial Intelligence, IUST)
- Dr. M. Abdul Jawad (M.Tech Student, Centre for Artificial Intelligence, IUST)

Affiliation: Centre for Artificial Intelligence, Islamic University of Science and Technology (IUST)
Copyright (c) 2026 Dar Rehan Rasool (@darrehanrasool). All rights reserved.
"""

import sys
from pathlib import Path

# Add project root to sys.path for robust cross-directory execution
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from university.main import main

if __name__ == '__main__':
    main()
