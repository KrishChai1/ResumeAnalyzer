"""
Resume Parser REST API
======================
FastAPI wrapper with PDF/Word upload and Swagger UI.

Run: uvicorn api_server:app --reload --port 8000
Swagger: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum
import json
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_parser_mcp import (
    parse_resume_full, ParseResumeInput, ResponseFormat,
    parse_name, parse_date_range, extract_all_skills, normalize_text
)

# PDF/Word support
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    try:
        import PyPDF2
        PDF_SUPPORT = True
    except ImportError:
        PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            raise HTTPException(400, f"PDF extraction failed: {e}")
    
    if not text.strip():
        raise HTTPException(400, "No text extracted from PDF (might be scanned/image)")
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = [c.text.strip() for c in row.cells if c.text.strip()]
                if row_text:
                    parts.append(" | ".join(row_text))
        text = "\n".join(parts)
        if not text.strip():
            raise HTTPException(400, "No text extracted from Word document")
        return text.strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Word extraction failed: {e}")


# ============================================================================
# API SETUP
# ============================================================================

app = FastAPI(
    title="Resume Parser API",
    description="""
## Production-Grade Resume Parser

### Features
- **📄 File Upload**: PDF and Word (.docx) support
- **Name Parsing**: firstname, lastname extraction
- **Skill Categorization**: Data Engineering, Programming, Cloud, DevOps, etc.
- **Experience Duration**: Months per skill calculation
- **Tools Extraction**: Per-job tool detection

### Output Format
Matches enterprise system integration requirements with:
- `firstname`, `lastname`, `name`
- `technical_skills` (flat list)
- `key_skills` (categorized with experience_months)
- `experience` (with Employer, tools, responsibilities)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ParseTextRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Raw resume text")
    filename: Optional[str] = Field(default=None, description="Original filename")


class ParseNameRequest(BaseModel):
    name: str = Field(..., description="Full name to parse", example="Dr. John Smith Jr.")


class ParseDatesRequest(BaseModel):
    date_range: str = Field(..., description="Date range", example="January 2020 - Present")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    return {"status": "healthy", "service": "Resume Parser API", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "pdf_support": PDF_SUPPORT,
        "docx_support": DOCX_SUPPORT
    }


@app.post("/parse/file", tags=["Parsing"], summary="Parse Resume from PDF/Word File")
async def parse_file(
    file: UploadFile = File(..., description="Resume file (PDF or .docx)")
):
    """
    Upload and parse a resume from PDF or Word document.
    
    **Supported formats:** PDF (.pdf), Word (.docx)
    
    **Returns:** Structured JSON with firstname, lastname, skills, experience, etc.
    """
    filename = file.filename.lower() if file.filename else ""
    
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        raise HTTPException(400, "Unsupported format. Use PDF or Word (.docx)")
    
    try:
        file_bytes = await file.read()
    except:
        raise HTTPException(400, "Failed to read file")
    
    if len(file_bytes) == 0:
        raise HTTPException(400, "File is empty")
    
    if filename.endswith('.pdf'):
        if not PDF_SUPPORT:
            raise HTTPException(500, "PDF support unavailable. Install: pip install pdfplumber")
        resume_text = extract_text_from_pdf(file_bytes)
    else:
        if not DOCX_SUPPORT:
            raise HTTPException(500, "Word support unavailable. Install: pip install python-docx")
        resume_text = extract_text_from_docx(file_bytes)
    
    try:
        result = await parse_resume_full(ParseResumeInput(
            resume_text=resume_text,
            filename=file.filename
        ))
        return json.loads(result)
    except Exception as e:
        raise HTTPException(400, f"Parse failed: {e}")


@app.post("/parse", tags=["Parsing"], summary="Parse Resume from Text")
async def parse_text(request: ParseTextRequest):
    """
    Parse resume from raw text input.
    
    **Returns:** Structured JSON matching enterprise format.
    """
    try:
        result = await parse_resume_full(ParseResumeInput(
            resume_text=request.resume_text,
            filename=request.filename
        ))
        return json.loads(result)
    except Exception as e:
        raise HTTPException(400, f"Parse failed: {e}")


@app.post("/parse/name", tags=["Utilities"], summary="Parse Person Name")
async def parse_person_name(request: ParseNameRequest):
    """Parse name into firstname, lastname, prefix, suffix."""
    parsed = parse_name(request.name)
    return {
        "firstname": parsed.first_name,
        "lastname": parsed.last_name,
        "full_name": parsed.full_name,
        "prefix": parsed.prefix,
        "suffix": parsed.suffix,
        "confidence": parsed.confidence
    }


@app.post("/parse/dates", tags=["Utilities"], summary="Parse Date Range")
async def parse_dates(request: ParseDatesRequest):
    """Parse date range and calculate duration in months."""
    dr = parse_date_range(request.date_range)
    if not dr:
        raise HTTPException(400, "Could not parse date range")
    return {
        "start_date": dr.start.to_date_string(),
        "end_date": dr.end.to_date_string(),
        "duration_months": dr.duration_months,
        "is_current": dr.is_current
    }


@app.post("/extract/skills", tags=["Utilities"], summary="Extract Skills from Text")
async def extract_skills(text: str = Field(..., min_length=10)):
    """Extract technical skills from text."""
    skills = extract_all_skills(text)
    return {"skills": skills, "count": len(skills)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
