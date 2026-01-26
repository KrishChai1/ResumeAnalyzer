"""
Resume Parser API Server
========================
FastAPI server for resume parsing with Swagger UI.
"""

import os
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from resume_parser_mcp import (
    parse_resume_full, ParseResumeInput, ResponseFormat,
    extract_technical_skills, normalize_text, ANTHROPIC_API_KEY
)

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


class ParseTextRequest(BaseModel):
    text: str = Field(..., min_length=50)
    use_ai_validation: bool = Field(default=True)
    filename: Optional[str] = Field(default=None)


class ExtractSkillsRequest(BaseModel):
    text: str = Field(..., min_length=10)


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Resume Parser API",
        "version": "2.0.0",
        "docs": "/docs",
        "ai_validation": "enabled" if ANTHROPIC_API_KEY else "disabled"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "ai_available": bool(ANTHROPIC_API_KEY)}


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(...),
    use_ai_validation: bool = Form(default=True)
):
    """Parse resume from PDF/Word file."""
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        raise HTTPException(400, f"Unsupported: {ext}")
    
    try:
        content = await file.read()
        
        if ext == 'pdf':
            text = extract_pdf(content)
        elif ext in ['docx', 'doc']:
            text = extract_docx(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        if len(text.strip()) < 50:
            raise HTTPException(400, "Insufficient text extracted")
        
        result = await parse_resume_full(ParseResumeInput(
            resume_text=text,
            filename=file.filename,
            use_ai_validation=use_ai_validation
        ))
        
        import json
        return json.loads(result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/parse", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """Parse resume from text."""
    try:
        result = await parse_resume_full(ParseResumeInput(
            resume_text=request.text,
            filename=request.filename,
            use_ai_validation=request.use_ai_validation
        ))
        import json
        return json.loads(result)
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/extract/skills", tags=["Utilities"])
async def extract_skills(request: ExtractSkillsRequest):
    """Extract technical skills from text."""
    skills = extract_technical_skills(request.text)
    return {"skills": skills, "count": len(skills)}


def extract_pdf(content: bytes) -> str:
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return '\n'.join([p.extract_text() or '' for p in pdf.pages])
    except ImportError:
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        return '\n'.join([p.extract_text() or '' for p in reader.pages])


def extract_docx(content: bytes) -> str:
    from docx import Document
    import io
    doc = Document(io.BytesIO(content))
    return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
