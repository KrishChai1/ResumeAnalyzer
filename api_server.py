"""
Resume Parser REST API
======================
FastAPI wrapper exposing MCP tools as REST endpoints with Swagger UI.

Run locally: uvicorn api_server:app --reload --port 8000
Swagger UI:  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import sys
import os
import re
import io

# Add the module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_parser_mcp import (
    parse_resume_full, ParseResumeInput,
    extract_skills_advanced,
    parse_name, parse_date_range,
    calculate_skill_durations, extract_experiences,
    normalize_skill_name, ResponseFormat
)

# PDF and Word extraction libraries
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
    """Extract text from PDF file."""
    text = ""
    try:
        # Try pdfplumber first (better quality)
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except:
        # Fallback to PyPDF2
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from PDF. The file might be scanned/image-based.")
    
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from Word document."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)
        
        # Also extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    text_parts.append(" | ".join(row_text))
        
        text = "\n".join(text_parts)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from Word document.")
        
        return text.strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from Word document: {str(e)}")

# ============================================================================
# API SETUP
# ============================================================================

app = FastAPI(
    title="Resume Parser API",
    description="""
## Production-Grade Resume Parser API

Enterprise-level resume parsing with Google/LinkedIn quality standards.

### Features
- **📄 File Upload**: Upload PDF or Word documents directly
- **Intelligent Name Parsing**: First, middle, last, prefix, suffix with confidence scores
- **Robust Date Extraction**: Multiple formats with accurate duration calculation
- **500+ Skill Taxonomy**: Comprehensive skill detection across 20+ categories
- **Experience Duration**: Calculate years of experience per skill
- **Job Matching**: Match resumes to job descriptions with gap analysis

### Endpoints
- `/parse/file` - **Upload PDF or Word file** ⭐
- `/parse` - Parse from raw text
- `/parse/name` - Parse person name
- `/parse/dates` - Parse date range and calculate duration
- `/extract/skills` - Extract and categorize skills
- `/match/job` - Match resume to job description
- `/validate` - Validate parsed resume JSON
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ResponseFormatEnum(str, Enum):
    json = "json"
    markdown = "markdown"


class ParseResumeRequest(BaseModel):
    """Request model for full resume parsing."""
    resume_text: str = Field(
        ..., 
        min_length=50,
        description="Raw text content of the resume",
        example="""JOHN DOE
Senior Software Engineer
San Francisco, CA | john.doe@email.com | (555) 123-4567

EXPERIENCE
Google - Senior Software Engineer
January 2020 - Present
- Led development of ML pipelines using Python and TensorFlow
- Architected microservices on AWS using Docker and Kubernetes

Amazon - Software Engineer
June 2017 - December 2019
- Built recommendation systems using Python and PyTorch
- Implemented data pipelines with Apache Spark

EDUCATION
MS Computer Science - Stanford University (2017)
BS Computer Science - UC Berkeley (2015)

SKILLS
Python, Java, TensorFlow, PyTorch, AWS, Docker, Kubernetes, SQL
"""
    )
    response_format: ResponseFormatEnum = Field(
        default=ResponseFormatEnum.json,
        description="Output format: 'json' or 'markdown'"
    )
    include_raw_sections: bool = Field(
        default=False,
        description="Include detected raw sections in output"
    )
    calculate_skill_durations: bool = Field(
        default=True,
        description="Calculate experience duration per skill"
    )


class ParseNameRequest(BaseModel):
    """Request model for name parsing."""
    name: str = Field(
        ...,
        description="Full name string to parse",
        example="Dr. John Michael Smith Jr."
    )


class ParseDatesRequest(BaseModel):
    """Request model for date parsing."""
    date_range: str = Field(
        ...,
        description="Date range string to parse",
        example="January 2020 - Present"
    )


class ExtractSkillsRequest(BaseModel):
    """Request model for skill extraction."""
    text: str = Field(
        ...,
        min_length=10,
        description="Text to extract skills from",
        example="Experienced in Python, TensorFlow, AWS, Docker, and Kubernetes. Built ML pipelines and microservices."
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for skills"
    )


class MatchJobRequest(BaseModel):
    """Request model for job matching."""
    resume_json: Dict[str, Any] = Field(
        ...,
        description="Parsed resume JSON (from /parse endpoint)"
    )
    job_description: str = Field(
        ...,
        min_length=50,
        description="Job description text",
        example="""Senior ML Engineer

Requirements:
- 5+ years experience in Machine Learning
- Expertise in Python, TensorFlow, PyTorch
- Cloud platform experience (AWS, GCP)
- Experience with Docker and Kubernetes
"""
    )


class ValidateResumeRequest(BaseModel):
    """Request model for resume validation."""
    resume_json: Dict[str, Any] = Field(
        ...,
        description="Parsed resume JSON to validate"
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Resume Parser API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "api": "up",
            "parser": "up"
        }
    }


@app.post("/parse", tags=["Parsing"], summary="Parse Complete Resume")
async def parse_resume(request: ParseResumeRequest):
    """
    Parse a complete resume with production-grade extraction.
    
    **Features:**
    - Intelligent name parsing (first, middle, last, suffix)
    - Robust date extraction and duration calculation
    - Multi-strategy skill extraction with 500+ skills taxonomy
    - Experience duration per skill calculation
    - Section detection and structured output
    - Confidence scores for extracted data
    
    **Returns:** Structured resume data with all components
    """
    try:
        response_format = ResponseFormat.MARKDOWN if request.response_format == "markdown" else ResponseFormat.JSON
        
        result = await parse_resume_full(ParseResumeInput(
            resume_text=request.resume_text,
            response_format=response_format,
            include_raw_sections=request.include_raw_sections,
            calculate_skill_durations=request.calculate_skill_durations
        ))
        
        if response_format == ResponseFormat.JSON:
            return json.loads(result)
        else:
            return {"markdown": result}
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/parse/file", tags=["Parsing"], summary="Parse Resume from PDF or Word File")
async def parse_resume_file(
    file: UploadFile = File(..., description="Resume file (PDF or Word .docx)"),
    calculate_skill_durations: bool = True
):
    """
    Upload and parse a resume from PDF or Word document.
    
    **Supported formats:**
    - PDF (.pdf)
    - Word Document (.docx)
    
    **Features:**
    - Automatic text extraction from documents
    - Intelligent name parsing (first, middle, last, suffix)
    - Robust date extraction and duration calculation
    - 500+ skill taxonomy with experience duration per skill
    - Confidence scores for extracted data
    
    **Returns:** Structured resume data with all components
    """
    # Validate file type
    filename = file.filename.lower() if file.filename else ""
    
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload a PDF (.pdf) or Word (.docx) file."
        )
    
    # Read file content
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    
    # Extract text based on file type
    if filename.endswith('.pdf'):
        if not PDF_SUPPORT:
            raise HTTPException(
                status_code=500, 
                detail="PDF support not available. Please install: pip install pdfplumber PyPDF2"
            )
        resume_text = extract_text_from_pdf(file_bytes)
    
    elif filename.endswith('.docx'):
        if not DOCX_SUPPORT:
            raise HTTPException(
                status_code=500, 
                detail="Word document support not available. Please install: pip install python-docx"
            )
        resume_text = extract_text_from_docx(file_bytes)
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    
    # Parse the extracted text
    try:
        result = await parse_resume_full(ParseResumeInput(
            resume_text=resume_text,
            response_format=ResponseFormat.JSON,
            include_raw_sections=False,
            calculate_skill_durations=calculate_skill_durations
        ))
        
        parsed_data = json.loads(result)
        
        # Add metadata about the source file
        parsed_data['source_file'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'text_length': len(resume_text)
        }
        
        return parsed_data
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse resume: {str(e)}")


@app.post("/parse/name", tags=["Parsing"], summary="Parse Person Name")
async def parse_person_name(request: ParseNameRequest):
    """
    Parse a name string into structured components.
    
    **Handles:**
    - Prefixes (Dr., Mr., Mrs., Prof.)
    - Suffixes (Jr., Sr., PhD, MBA, III)
    - Middle names
    - Compound last names (van, von, de, etc.)
    
    **Returns:** Structured name with confidence score
    """
    try:
        parsed = parse_name(request.name)
        return {
            "full_name": parsed.full_name,
            "first_name": parsed.first_name,
            "middle_name": parsed.middle_name,
            "last_name": parsed.last_name,
            "prefix": parsed.prefix,
            "suffix": parsed.suffix,
            "confidence": parsed.confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/parse/dates", tags=["Parsing"], summary="Parse Date Range")
async def parse_date_range_endpoint(request: ParseDatesRequest):
    """
    Parse a date range string and calculate duration.
    
    **Handles formats:**
    - "Jan 2020 - Present"
    - "January 2020 to December 2023"
    - "01/2020 - 12/2023"
    - "2020 - 2023"
    
    **Returns:** Parsed dates with duration in months/years
    """
    try:
        date_range = parse_date_range(request.date_range)
        
        if not date_range:
            raise HTTPException(status_code=400, detail="Could not parse date range")
        
        return {
            "start_date": {
                "year": date_range.start.year,
                "month": date_range.start.month,
                "original": date_range.start.original
            },
            "end_date": {
                "year": date_range.end.year,
                "month": date_range.end.month,
                "is_present": date_range.end.is_present,
                "original": date_range.end.original
            },
            "duration": {
                "months": date_range.duration_months,
                "years": date_range.duration_years
            },
            "is_current": date_range.is_current
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/extract/skills", tags=["Extraction"], summary="Extract Skills from Text")
async def extract_skills_endpoint(request: ExtractSkillsRequest):
    """
    Extract and categorize skills from text.
    
    **Uses a comprehensive taxonomy of 500+ skills across categories:**
    - Programming languages
    - Frontend/Backend frameworks
    - Cloud platforms (AWS, Azure, GCP services)
    - Databases and data warehouses
    - DevOps and infrastructure
    - AI/ML and data science
    - And more...
    
    **Returns:** Categorized skills with confidence scores
    """
    try:
        skills = extract_skills_advanced(request.text)
        
        # Filter by confidence and format
        result = {}
        total_count = 0
        
        for category, skill_list in skills.items():
            filtered = [s for s in skill_list if s.confidence >= request.min_confidence]
            if filtered:
                result[category] = [
                    {
                        "name": s.name,
                        "normalized": s.normalized,
                        "confidence": s.confidence
                    }
                    for s in filtered
                ]
                total_count += len(filtered)
        
        return {
            "total_skills": total_count,
            "categories_found": list(result.keys()),
            "skills_by_category": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/match/job", tags=["Analysis"], summary="Match Resume to Job")
async def match_job_endpoint(request: MatchJobRequest):
    """
    Analyze how well a resume matches a job description.
    
    **Provides:**
    - Overall match score (percentage)
    - Matched skills with experience levels
    - Missing required skills
    - Experience gap analysis
    - Actionable recommendations
    
    **Returns:** Match analysis with score and gaps
    """
    try:
        resume = request.resume_json
        job_text = request.job_description
        
        # Extract skills from job description
        job_skills = extract_skills_advanced(job_text)
        job_skills_flat = set()
        for cat_skills in job_skills.values():
            job_skills_flat.update(s.normalized for s in cat_skills)
        
        # Get resume skills
        resume_skills_flat = set()
        for cat_skills in resume.get('skills', {}).values():
            for skill in cat_skills:
                name = skill.get('name', skill) if isinstance(skill, dict) else skill
                resume_skills_flat.add(normalize_skill_name(name))
        
        # Calculate matches
        matched = job_skills_flat.intersection(resume_skills_flat)
        missing = job_skills_flat - resume_skills_flat
        
        # Match score
        match_score = (len(matched) / len(job_skills_flat) * 100) if job_skills_flat else 0
        
        # Experience check
        years_pattern = r'(\d+)\+?\s*years?'
        year_match = re.search(years_pattern, job_text, re.IGNORECASE)
        required_years = int(year_match.group(1)) if year_match else None
        
        actual_years = resume.get('total_experience', {}).get('years', 0)
        experience_match = actual_years >= required_years if required_years else True
        
        # Recommendations
        recommendations = []
        if missing:
            top_missing = list(missing)[:5]
            recommendations.append(f"Consider highlighting or developing: {', '.join(top_missing)}")
        
        if not experience_match and required_years:
            recommendations.append(f"Role requires {required_years}+ years; you have {actual_years}")
        
        if match_score >= 75:
            recommendations.append("Strong match! Tailor resume to emphasize matched skills.")
        elif match_score >= 50:
            recommendations.append("Good potential. Emphasize transferable skills.")
        else:
            recommendations.append("Consider building required skills before applying.")
        
        return {
            "match_score": round(match_score, 1),
            "skill_coverage": f"{len(matched)}/{len(job_skills_flat)}",
            "matched_skills": sorted(list(matched)),
            "missing_skills": sorted(list(missing)),
            "experience": {
                "required": f"{required_years}+ years" if required_years else "Not specified",
                "actual": f"{actual_years} years",
                "meets_requirement": experience_match
            },
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate", tags=["Analysis"], summary="Validate Resume JSON")
async def validate_resume_endpoint(request: ValidateResumeRequest):
    """
    Validate parsed resume JSON and identify issues.
    
    **Checks for:**
    - Character encoding issues
    - Missing required fields (name, contact, etc.)
    - Invalid date formats
    - Skills with 0 experience when they should have more
    - Data consistency
    
    **Returns:** Validation report with issues and warnings
    """
    try:
        resume = request.resume_json
        issues = []
        warnings = []
        
        # Check encoding
        json_str = json.dumps(resume, default=str)
        encoding_problems = ['â€"', 'â€™', 'â€œ', 'Ã', 'Â']
        for problem in encoding_problems:
            if problem in json_str:
                issues.append(f"Encoding issue detected: characters like {repr(problem)}")
                break
        
        # Check name
        name = resume.get('name', {})
        if isinstance(name, dict):
            if not name.get('first_name'):
                issues.append("Missing first_name in name object")
            if not name.get('last_name'):
                warnings.append("Missing last_name (might be single name)")
        elif isinstance(name, str):
            warnings.append("Name should be a structured object with first_name/last_name")
        else:
            issues.append("Missing name field")
        
        # Check contact
        contact = resume.get('contact', {})
        if not contact.get('email'):
            warnings.append("No email found")
        if not contact.get('phone'):
            warnings.append("No phone found")
        
        # Check experience
        experience = resume.get('experience', [])
        if not experience:
            warnings.append("No work experience extracted")
        else:
            for i, exp in enumerate(experience):
                dr = exp.get('date_range', {})
                if dr.get('duration_months', 0) == 0:
                    if dr.get('start_date') and dr.get('end_date'):
                        issues.append(f"Experience #{i+1}: duration is 0 months but dates exist")
        
        # Check skills
        skills = resume.get('skills', {})
        if not skills:
            warnings.append("No skills extracted")
        
        return {
            "is_valid": len(issues) == 0,
            "issues_count": len(issues),
            "warnings_count": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "field_status": {
                "has_name": bool(name),
                "has_contact": bool(contact),
                "has_experience": bool(experience),
                "has_skills": bool(skills),
                "has_education": bool(resume.get('education'))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
