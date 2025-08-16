#!/usr/bin/env python3
"""
Quantum Batch Worker Launcher Script

This script starts the batch worker process for processing quantum transformation jobs.
Can be run standalone or via cron for scheduled processing.
"""

import sys
import os

# Add the API directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

from core.batch_worker import main

if __name__ == "__main__":
    main()