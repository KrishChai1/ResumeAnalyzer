"""
Resume Parser API Server v4.0
=============================
Production-ready FastAPI server using proven multi-format parser.

Endpoints:
- GET / - API info
- GET /health - Health check
- GET /docs - Swagger UI
- POST /parse/file - Parse uploaded resume
- POST /parse/text - Parse resume text
"""

import os
import io
import json
import asyncio
import re
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import proven parser
from resume_parser_mcp import (
    parse_resume_full, 
    ParseResumeInput, 
    normalize_text,
    extract_text_from_docx_with_tables,
    extract_all_text_from_docx,
    ANTHROPIC_API_KEY
)

app = FastAPI(
    title="Resume Parser API",
    description="""
## Enterprise Resume Parser v4.0

### Architecture:
- **Multi-Strategy Extraction**: 12+ pattern-based strategies
- **AI Enhancement**: Claude API for complex cases
- **Validation Agent**: Quality scoring & completeness checks

### Supported Formats:
- PDF (pdfplumber + PyPDF2 fallback)
- DOCX (paragraphs + tables + text boxes)
- TXT

### Key Features:
- Handles pipe, dash, table, worked-as formats
- Text box extraction for sidebar layouts
- Table extraction for structured resumes
- Automatic duration calculation
- Skill-to-experience mapping
    """,
    version="4.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MODELS
# =============================================================================

class ParseTextRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Resume text content")
    use_ai: bool = Field(default=True, description="Enable AI validation")
    filename: Optional[str] = Field(default=None, description="Original filename")


class HealthResponse(BaseModel):
    status: str
    ai_available: bool
    version: str


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF using pdfplumber with PyPDF2 fallback."""
    text = ""
    
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
            if text.strip():
                return text
    except Exception:
        pass
    
    # Fallback to PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = '\n'.join([page.extract_text() or '' for page in reader.pages])
    except Exception as e:
        raise HTTPException(500, f"PDF extraction failed: {str(e)}")
    
    return text


def extract_text_from_docx(content: bytes, save_path: str = None) -> str:
    """Extract text from DOCX including tables and text boxes."""
    from docx import Document
    
    doc = Document(io.BytesIO(content))
    all_text = []
    
    # 1. Extract paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            all_text.append(text)
    
    # 2. Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                all_text.append(' | '.join(row_text))
    
    # 3. Extract text boxes (for sidebar/multi-column layouts)
    try:
        xml_str = doc.element.xml
        pattern = r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>'
        for match in re.findall(pattern, xml_str, re.DOTALL):
            text_pattern = r'<w:t[^>]*>([^<]+)</w:t>'
            texts = re.findall(text_pattern, match)
            if texts:
                combined = ' '.join(texts)
                if combined.strip() and len(combined) > 5:
                    all_text.append(combined)
    except Exception:
        pass
    
    return '\n'.join(all_text)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "Resume Parser API",
        "version": "4.0.0",
        "features": [
            "Multi-format support (PDF, DOCX, TXT)",
            "12+ extraction patterns",
            "AI enhancement (optional)",
            "Table & text box extraction",
            "Validation scoring"
        ],
        "ai_enabled": bool(ANTHROPIC_API_KEY),
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ai_available": bool(ANTHROPIC_API_KEY),
        "version": "4.0.0"
    }


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, TXT)"),
    use_ai: bool = Form(default=True, description="Enable AI validation")
):
    """
    Parse resume from uploaded file.
    
    Supports: PDF, DOCX, DOC, TXT
    
    Returns structured data including:
    - Contact info (name, email, phone, linkedin)
    - Work experience with responsibilities
    - Education history
    - Certifications
    - Technical skills with experience mapping
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        raise HTTPException(400, f"Unsupported format: {ext}. Use PDF, DOCX, or TXT")
    
    try:
        content = await file.read()
        
        # Extract text based on file type
        if ext == 'pdf':
            text = extract_text_from_pdf(content)
        elif ext in ['docx', 'doc']:
            text = extract_text_from_docx(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        if len(text.strip()) < 50:
            raise HTTPException(400, "Insufficient text extracted from file")
        
        # Parse using proven multi-strategy parser
        result = await parse_resume_full(ParseResumeInput(
            resume_text=text,
            filename=file.filename,
            use_ai_validation=use_ai and bool(ANTHROPIC_API_KEY)
        ))
        
        return json.loads(result)
        
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Parsing error: {str(e)}")


@app.post("/parse/text", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """
    Parse resume from raw text.
    
    Useful for:
    - Text already extracted from documents
    - Copy-pasted resume content
    - Testing and debugging
    """
    try:
        result = await parse_resume_full(ParseResumeInput(
            resume_text=request.text,
            filename=request.filename or "text_input",
            use_ai_validation=request.use_ai and bool(ANTHROPIC_API_KEY)
        ))
        
        return json.loads(result)
        
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(500, f"Parsing error: {str(e)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
