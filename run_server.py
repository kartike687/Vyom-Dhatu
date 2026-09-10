"""
MOIL Manganese Mine Intelligence & Digital Twin Launcher
Starts the FastAPI server on port 8000 and prints connection details.
"""

import sys
import os
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    print("=" * 70)
    print("  MOIL MANGANESE MINE INTELLIGENCE & DIGITAL TWIN SIMULATOR")
    print("  Smart India Hackathon 2026 Problem Statement")
    print("=" * 70)
    print("  -> Offline-first AI Mining Command Centre")
    print("  -> Local URL: http://127.0.0.1:8000")
    print("  -> Swagger API Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)
