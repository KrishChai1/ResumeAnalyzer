"""
Resume Parser API Server
========================
FastAPI server for resume parsing with Swagger UI.
Enhanced with comprehensive logging for Railway deployment.
"""

import os
import sys
import logging
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================================
# LOGGING CONFIGURATION - Must be first
# ============================================================================

# Force stdout to be unbuffered for Railway
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("resume_parser_api")
logger.setLevel(logging.DEBUG)

# Log startup
logger.info("=" * 60)
logger.info("RESUME PARSER API - STARTING")
logger.info("=" * 60)
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PORT env: {os.environ.get('PORT', 'not set')}")
logger.info(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")

# ============================================================================
# IMPORT PARSER MODULE
# ============================================================================

try:
    logger.info("Importing resume_parser_mcp module...")
    from resume_parser_mcp import (
        parse_resume_full, ParseResumeInput, ResponseFormat,
        extract_technical_skills, normalize_text, ANTHROPIC_API_KEY
    )
    logger.info("Successfully imported resume_parser_mcp")
except ImportError as e:
    logger.error(f"Failed to import resume_parser_mcp: {e}")
    logger.error(traceback.format_exc())
    raise
except Exception as e:
    logger.error(f"Unexpected error importing resume_parser_mcp: {e}")
    logger.error(traceback.format_exc())
    raise

# ============================================================================
# FASTAPI APP
# ============================================================================

logger.info("Creating FastAPI app...")

app = FastAPI(
    title="Resume Parser API",
    description="""
## Production-Grade Resume Parser with AI Validation

### Features:
- Multi-format: PDF, DOCX, TXT
- Intelligent extraction: Name, contact, education, experience, skills
- Skill categorization with experience months
- Optional Claude AI validation

### Configuration:
Set `ANTHROPIC_API_KEY` environment variable for AI enhancement.

### Endpoints:
- `GET /` - API info
- `GET /health` - Health check
- `POST /parse/file` - Parse uploaded file
- `POST /parse` - Parse text
- `POST /extract/skills` - Extract skills from text
    """,
    version="2.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI app created successfully")

# ============================================================================
# REQUEST MODELS
# ============================================================================

class ParseTextRequest(BaseModel):
    text: str = Field(..., min_length=50)
    use_ai_validation: bool = Field(default=True)
    filename: Optional[str] = Field(default=None)


class ExtractSkillsRequest(BaseModel):
    text: str = Field(..., min_length=10)


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("APPLICATION STARTUP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"AI Validation: {'ENABLED' if ANTHROPIC_API_KEY else 'DISABLED'}")
    logger.info("Ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    logger.info("GET / - Root endpoint called")
    return {
        "name": "Resume Parser API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "ai_validation": "enabled" if ANTHROPIC_API_KEY else "disabled"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("GET /health - Health check called")
    return {
        "status": "healthy",
        "ai_available": bool(ANTHROPIC_API_KEY),
        "version": "2.0.0"
    }


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(...),
    use_ai_validation: bool = Form(default=True)
):
    """Parse resume from PDF/Word file."""
    logger.info(f"POST /parse/file - Received file: {file.filename}")
    logger.info(f"  Content-Type: {file.content_type}")
    logger.info(f"  AI Validation: {use_ai_validation}")
    
    if not file.filename:
        logger.warning("No filename provided")
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.lower().split('.')[-1]
    logger.info(f"  Extension: {ext}")
    
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        logger.warning(f"Unsupported file extension: {ext}")
        raise HTTPException(400, f"Unsupported file type: {ext}")
    
    try:
        content = await file.read()
        logger.info(f"  File size: {len(content)} bytes")
        
        # Extract text based on file type
        if ext == 'pdf':
            logger.info("  Extracting text from PDF...")
            text = extract_pdf(content)
        elif ext in ['docx', 'doc']:
            logger.info("  Extracting text from DOCX...")
            text = extract_docx(content)
        else:
            logger.info("  Reading as plain text...")
            text = content.decode('utf-8', errors='ignore')
        
        logger.info(f"  Extracted text length: {len(text)} chars")
        
        if len(text.strip()) < 50:
            logger.warning(f"Insufficient text extracted: {len(text.strip())} chars")
            raise HTTPException(400, "Insufficient text extracted from file")
        
        # Parse resume
        logger.info("  Parsing resume...")
        result = await parse_resume_full(ParseResumeInput(
            resume_text=text,
            filename=file.filename,
            use_ai_validation=use_ai_validation
        ))
        
        logger.info("  Parse complete, returning JSON")
        import json
        return json.loads(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error parsing file: {str(e)}")


@app.post("/parse", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """Parse resume from text."""
    logger.info(f"POST /parse - Received text ({len(request.text)} chars)")
    logger.info(f"  AI Validation: {request.use_ai_validation}")
    logger.info(f"  Filename: {request.filename}")
    
    try:
        logger.info("  Parsing resume...")
        result = await parse_resume_full(ParseResumeInput(
            resume_text=request.text,
            filename=request.filename,
            use_ai_validation=request.use_ai_validation
        ))
        
        logger.info("  Parse complete, returning JSON")
        import json
        return json.loads(result)
    except Exception as e:
        logger.error(f"Error parsing text: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error parsing text: {str(e)}")


@app.post("/extract/skills", tags=["Utilities"])
async def extract_skills(request: ExtractSkillsRequest):
    """Extract technical skills from text."""
    logger.info(f"POST /extract/skills - Received text ({len(request.text)} chars)")
    
    try:
        skills = extract_technical_skills(request.text)
        logger.info(f"  Extracted {len(skills)} skills")
        return {"skills": skills, "count": len(skills)}
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Error extracting skills: {str(e)}")


# ============================================================================
# FILE EXTRACTION HELPERS
# ============================================================================

def extract_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    logger.debug("extract_pdf called")
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages_text = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ''
                pages_text.append(page_text)
                logger.debug(f"  PDF page {i+1}: {len(page_text)} chars")
            return '\n'.join(pages_text)
    except ImportError:
        logger.warning("pdfplumber not available, trying PyPDF2")
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            pages_text.append(page_text)
            logger.debug(f"  PDF page {i+1}: {len(page_text)} chars")
        return '\n'.join(pages_text)


def extract_docx(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    logger.debug("extract_docx called")
    from docx import Document
    import io
    doc = Document(io.BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    logger.debug(f"  Extracted {len(paragraphs)} paragraphs")
    return '\n'.join(paragraphs)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info("=" * 60)
    logger.info(f"STARTING UVICORN SERVER")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info("=" * 60)
    
    # Run with detailed logging
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
