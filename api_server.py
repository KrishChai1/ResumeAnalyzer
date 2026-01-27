"""
Resume Parser API Server v3.2
=============================
FastAPI server with intelligent AI-first parsing + regex fallback.
"""

import os
import asyncio
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

app = FastAPI(
    title="Resume Parser API",
    description="""
## Intelligent Resume Parser with AI-First Approach

### Features:
- **AI-First**: Uses Claude AI for intelligent extraction (any format)
- **Regex Fallback**: Reliable backup when AI unavailable
- **Multi-format**: PDF, DOCX, TXT support
- **Enterprise Grade**: Handles complex layouts, tables, text boxes

### Configuration:
Set `ANTHROPIC_API_KEY` environment variable for AI-powered parsing.
    """,
    version="3.2.0",
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
    use_ai: bool = Field(default=True)
    filename: Optional[str] = Field(default=None)


class ExtractSkillsRequest(BaseModel):
    text: str = Field(..., min_length=10)


@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Resume Parser API",
        "version": "3.2.0",
        "docs": "/docs",
        "ai_enabled": bool(ANTHROPIC_API_KEY),
        "mode": "AI-First" if ANTHROPIC_API_KEY else "Regex-Only"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "ai_available": bool(ANTHROPIC_API_KEY),
        "parser_mode": "intelligent" if ANTHROPIC_API_KEY else "regex"
    }


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(...),
    use_ai: bool = Form(default=True)
):
    """
    Parse resume from PDF/DOCX file.
    
    - **file**: Resume file (PDF, DOCX, TXT)
    - **use_ai**: Enable AI-powered parsing (default: true)
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = file.filename.lower().split('.')[-1]
    if ext not in ['pdf', 'docx', 'doc', 'txt']:
        raise HTTPException(400, f"Unsupported format: {ext}")
    
    try:
        content = await file.read()
        
        # Save to temp file for enhanced extraction
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extract text based on format
            if ext == 'pdf':
                text = extract_pdf_text(tmp_path)
            elif ext in ['docx', 'doc']:
                text = extract_docx_text(tmp_path)
            else:
                text = content.decode('utf-8', errors='ignore')
            
            if len(text.strip()) < 50:
                raise HTTPException(400, "Insufficient text extracted from file")
            
            # Parse using appropriate method
            if use_ai and ANTHROPIC_API_KEY:
                from intelligent_parser import parse_resume_intelligent
                result = await parse_resume_intelligent(
                    text=text,
                    filename=file.filename,
                    file_path=tmp_path
                )
            else:
                from resume_parser_mcp import parse_resume_full, ParseResumeInput, normalize_text
                text = normalize_text(text)
                result = await parse_resume_full(ParseResumeInput(
                    resume_text=text,
                    filename=file.filename,
                    file_path=tmp_path,
                    use_ai_validation=False
                ))
            
            import json
            return json.loads(result)
        
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Parsing error: {str(e)}")


@app.post("/parse", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """
    Parse resume from plain text.
    
    - **text**: Resume text content
    - **use_ai**: Enable AI-powered parsing (default: true)
    """
    try:
        if request.use_ai and ANTHROPIC_API_KEY:
            from intelligent_parser import parse_resume_intelligent
            result = await parse_resume_intelligent(
                text=request.text,
                filename=request.filename or ""
            )
        else:
            from resume_parser_mcp import parse_resume_full, ParseResumeInput, normalize_text
            result = await parse_resume_full(ParseResumeInput(
                resume_text=normalize_text(request.text),
                filename=request.filename,
                use_ai_validation=False
            ))
        
        import json
        return json.loads(result)
    except Exception as e:
        raise HTTPException(500, f"Parsing error: {str(e)}")


def extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using multiple methods."""
    text = ""
    
    # Try pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = '\n'.join([p.extract_text() or '' for p in pdf.pages])
    except:
        pass
    
    # Fallback to PyPDF2
    if len(text.strip()) < 100:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = '\n'.join([p.extract_text() or '' for p in reader.pages])
        except:
            pass
    
    return text


def extract_docx_text(file_path: str) -> str:
    """Extract ALL text from DOCX including tables and text boxes."""
    from docx import Document
    doc = Document(file_path)
    
    all_text = []
    
    # Paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            all_text.append(para.text.strip())
    
    # Tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                all_text.append(' | '.join(row_text))
    
    # Text boxes
    try:
        for txbx in doc.element.iter():
            if txbx.tag.endswith('txbxContent'):
                texts = [t.text for t in txbx.iter() if t.tag.endswith('}t') and t.text]
                if texts:
                    content = ' '.join(texts)
                    if content not in all_text:
                        all_text.append(content)
    except:
        pass
    
    return '\n'.join(all_text)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 
