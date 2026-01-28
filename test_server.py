"""
Ultra-minimal test server - NO dependencies except FastAPI
"""
import os
import sys

print("=" * 60, flush=True)
print("STARTING TEST SERVER", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"PORT env: {os.environ.get('PORT', 'NOT SET')}", flush=True)
print("=" * 60, flush=True)

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok", "message": "Test server works!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
