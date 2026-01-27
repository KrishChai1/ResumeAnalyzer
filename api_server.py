"""
Resume Parser API Server v3.0
==============================
FastAPI server with Swagger UI using Smart Agentic Parser.
"""

import os
import io
import tempfile
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from smart_parser import SmartResumeParser, normalize_text

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(
    title="Smart Resume Parser API",
    description="""
## Enterprise-Grade Agentic Resume Parser v3.0

### Architecture:
- **Format Detector**: Auto-detects resume structure
- **Multi-Strategy Extraction**: Specialized extractors for each format
- **Validation Agent**: Quality scoring (0-100)
- **AI Fallback**: Claude API for complex cases

### Supported Formats:
- PIPE: `Title | Company | Date`
- SLASH_DATE: `MM/YYYY - MM/YYYY`
- ROLE_BASED: `Company Date` then `ROLE: Title`
- WORKED_AS: `Worked as X in Y from A to B`
- TABLE: Structured tables
- TEXTBOX: Multi-column layouts

### Configuration:
Set `ANTHROPIC_API_KEY` for AI enhancement.
    """,
    version="3.0.0",
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
        "name": "Smart Resume Parser API",
        "version": "3.0.0",
        "docs": "/docs",
        "ai_available": bool(ANTHROPIC_API_KEY),
        "supported_formats": ["pdf", "docx", "doc", "txt"]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "ai_available": bool(ANTHROPIC_API_KEY)}


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(...),
    use_ai_validation: bool = Form(default=True)
):
    """Parse resume from PDF/DOCX file."""
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        raise HTTPException(400, f"Unsupported format: {ext}")
    
    try:
        content = await file.read()
        
        # Extract text based on file type
        if ext == 'pdf':
            text = extract_pdf(content)
        elif ext in ['docx', 'doc']:
            text = extract_docx(content)
        else:
            text = content.decode('utf-8', errors='ignore')
        
        if len(text.strip()) < 50:
            raise HTTPException(400, "Insufficient text extracted from file")
        
        # Parse using smart parser
        result = await SmartResumeParser.parse(text, use_ai=use_ai_validation)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error parsing resume: {str(e)}")


@app.post("/parse", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """Parse resume from plain text."""
    try:
        result = await SmartResumeParser.parse(
            request.text, 
            use_ai=request.use_ai_validation
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


def extract_pdf(content: bytes) -> str:
    """Extract text from PDF with fallback."""
    text = ""
    
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
    except Exception:
        pass
    
    # Fallback to PyPDF2 if pdfplumber failed or returned minimal text
    if len(text.strip()) < 100:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = '\n'.join([page.extract_text() or '' for page in reader.pages])
        except Exception:
            pass
    
    return text


def extract_docx(content: bytes) -> str:
    """Extract text from DOCX including tables and text boxes."""
    from docx import Document
    
    doc = Document(io.BytesIO(content))
    all_text = []
    
    # Extract paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            all_text.append(text)
    
    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = ' | '.join([cell.text.strip() for cell in row.cells if cell.text.strip()])
            if row_text:
                all_text.append(row_text)
    
    # Extract text boxes (for multi-column layouts)
    try:
        import re as regex
        xml_str = doc.element.xml
        pattern = r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>'
        matches = regex.findall(pattern, xml_str, regex.DOTALL)
        
        for match in matches:
            text_pattern = r'<w:t[^>]*>([^<]+)</w:t>'
            texts = regex.findall(text_pattern, match)
            if texts:
                combined = ' '.join(texts)
                combined = ' '.join(combined.split())
                if combined and len(combined) > 5:
                    all_text.append(combined)
    except Exception:
        pass
    
    return '\n'.join(all_text)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
