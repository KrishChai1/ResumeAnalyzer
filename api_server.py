"""
Resume Parser API Server v3.1
==============================
FastAPI server with Swagger UI using Enterprise Resume Parser.
Includes Claude AI validation for complex/difficult resumes.
"""

import os
import io
import json
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from resume_parser_mcp import (
    parse_resume_full, ParseResumeInput, normalize_text, ANTHROPIC_API_KEY,
    extract_all_text_from_docx
)

app = FastAPI(
    title="Enterprise Resume Parser API",
    description="""
## Enterprise-Grade Agentic Resume Parser v3.1

### Architecture:
- **Multi-Strategy Extraction**: 11+ extraction patterns
- **Validation Agent**: Quality scoring (0-100)
- **AI Fallback**: Claude API automatically triggered for score < 60

### Supported Formats:
- PIPE: `Title | Company | Date`
- SLASH_DATE: `MM/YYYY - MM/YYYY` (Javvaji style)
- ROLE_BASED: `Company Date` then `ROLE: Title` (Sarwar style)
- CLIENT_DATE: `Client: Company – Location Date` (Naveen style)
- WORKED_AS: `Worked as X in Y from A to B`
- TABLE: Structured tables (Ramaswamy style)
- TEXTBOX: Multi-column layouts (Nageswara style)

### AI Validation:
Set `ANTHROPIC_API_KEY` environment variable to enable.
AI fallback triggers automatically when:
- Validation score < 60
- Name is missing or invalid
- Critical fields are missing
    """,
    version="3.1.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ParseTextRequest(BaseModel):
    text: str = Field(..., min_length=50)
    use_ai_validation: bool = Field(default=True)


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Enterprise Resume Parser API",
        "version": "3.1.0",
        "docs": "/docs",
        "ai_available": bool(ANTHROPIC_API_KEY),
        "ai_status": "enabled" if ANTHROPIC_API_KEY else "disabled (set ANTHROPIC_API_KEY)",
        "supported_formats": ["pdf", "docx", "doc", "txt"]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy", 
        "ai_available": bool(ANTHROPIC_API_KEY),
        "message": "AI validation ready" if ANTHROPIC_API_KEY else "AI validation disabled - set ANTHROPIC_API_KEY"
    }


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(...),
    use_ai_validation: bool = Form(default=True)
):
    """
    Parse resume from PDF/DOCX file.
    
    - **file**: Resume file (PDF, DOCX, DOC, or TXT)
    - **use_ai_validation**: Enable Claude AI fallback for difficult resumes (default: True)
    
    Returns structured JSON with:
    - parsed_resume: All extracted fields
    - validation_score: Quality score 0-100
    - validation_issues: List of any problems found
    - ai_enhanced: True if Claude AI was used to fix issues
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        raise HTTPException(400, f"Unsupported format: {ext}. Use PDF, DOCX, DOC, or TXT.")
    
    try:
        content = await file.read()
        file_path = None
        
        # Extract text based on file type
        if ext == 'pdf':
            text = extract_pdf(content)
        elif ext in ['docx', 'doc']:
            # Save to temp file for table/textbox extraction
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
                tmp.write(content)
                file_path = tmp.name
            
            # Extract text including tables and textboxes
            text = extract_all_text_from_docx(file_path)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        text = normalize_text(text)
        
        if len(text.strip()) < 50:
            raise HTTPException(400, "Insufficient text extracted from file. The file may be empty or corrupted.")
        
        # Parse using enterprise parser with AI fallback
        result = await parse_resume_full(ParseResumeInput(
            resume_text=text,
            filename=file.filename,
            file_path=file_path,
            use_ai_validation=use_ai_validation
        ))
        
        # Clean up temp file
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except:
                pass
        
        return json.loads(result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error parsing resume: {str(e)}")


@app.post("/parse", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """
    Parse resume from plain text.
    
    - **text**: Resume text (minimum 50 characters)
    - **use_ai_validation**: Enable Claude AI fallback (default: True)
    """
    try:
        result = await parse_resume_full(ParseResumeInput(
            resume_text=request.text,
            use_ai_validation=request.use_ai_validation
        ))
        return json.loads(result)
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


def extract_pdf(content: bytes) -> str:
    """Extract text from PDF with fallback."""
    text = ""
    
    # Try pdfplumber first (better formatting)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = '\n'.join([p.extract_text() or '' for p in pdf.pages])
    except Exception:
        pass
    
    # Fallback to PyPDF2 if pdfplumber fails or returns too little text
    if len(text.strip()) < 100:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = '\n'.join([p.extract_text() or '' for p in reader.pages])
        except Exception:
            pass
    
    return text


def extract_docx(content: bytes) -> str:
    """Extract text from DOCX."""
    from docx import Document
    doc = Document(io.BytesIO(content))
    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Enterprise Resume Parser API on port {port}")
    print(f"AI Validation: {'ENABLED' if ANTHROPIC_API_KEY else 'DISABLED (set ANTHROPIC_API_KEY)'}")
    uvicorn.run(app, host="0.0.0.0", port=port)
