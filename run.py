#!/usr/bin/env python3
"""
Application launcher.
Run this file to open the unified GUI for camera control and monitoring.
"""
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


if __name__ == "__main__":
    try:
        from app_gui import main

        main()
    except ImportError as e:
        print(f"Import failed: {e}")
        print("Please ensure all required project files exist in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Application failed: {e}")
        sys.exit(1)
