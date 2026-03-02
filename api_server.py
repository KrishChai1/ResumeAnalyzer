#!/usr/bin/env python3
"""
Resume Parser API Server v8.0.0
Main entry point for Railway deployment
"""

# Import everything from the parser module
from resume_parser_mcp import *

# This file just imports and runs the FastAPI app from resume_parser_mcp.py
# The app, routes, and all logic are in resume_parser_mcp.py

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8080))
    
    print(f"\n{'='*60}")
    print(f"RESUME PARSER KRISH API v{VERSION}")
    print(f"AI Model: {AI_MODEL if 'AI_MODEL' in dir() else 'claude-sonnet-4-20250514'}")
    print(f"AI Ready: {'Yes' if ANTHROPIC_API_KEY else 'No - Set ANTHROPIC_API_KEY'}")
    print(f"Docs: http://localhost:{port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
