#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         RESUME PARSER API v6.0 - COMPLETE AGENTIC FRAMEWORK                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  AGENTIC ARCHITECTURE:                                                        ║
║  =====================                                                        ║
║                                                                               ║
║  ┌────────────────────────────────────────────────────────────────────────┐  ║
║  │                        ORCHESTRATION LAYER                              │  ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  ║
║  │  │ EXTRACTION  │─▶│ VALIDATION  │─▶│ AI ENHANCE  │─▶│ OUTPUT      │   │  ║
║  │  │ AGENT       │  │ AGENT       │  │ AGENT       │  │ AGENT       │   │  ║
║  │  │ (7 Patterns)│  │ (Scoring)   │  │ (Claude)    │  │ (JSON)      │   │  ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │  ║
║  └────────────────────────────────────────────────────────────────────────┘  ║
║                                                                               ║
║  EXTRACTION PATTERNS:                                                         ║
║  ════════════════════                                                         ║
║  Pattern 1: Standard "Company – Title Date" format                            ║
║  Pattern 2: "Worked as X in Y from A to B" format (Madhuri)                   ║
║  Pattern 3: Table "Client: X | Duration: Y" format (Ramaswamy)                ║
║  Pattern 4: "Title (Date) – Client – Employer" format (Nageswara)             ║
║  Pattern 5: "Company Date" then "Title" next line (Khaliq)                    ║
║  Pattern 6: Pipe format "Title | Date" with company above (Jimmy)             ║
║  Pattern 7: "ROLE:" / "DESIGNATION:" keyword format (Sarwer)                  ║
║                                                                               ║
║  SUPPORTED FORMATS: PDF, DOCX (tables/textboxes), TXT, ZIP                    ║
║                                                                               ║
║  FEATURES:                                                                    ║
║  • Intelligent file type detection (magic bytes)                              ║
║  • Multi-strategy text extraction                                             ║
║  • 15+ skill categories with experience calculation                           ║
║  • AI-powered validation and enhancement (Claude API)                         ║
║  • Comprehensive error handling                                               ║
║  • FastAPI with Swagger UI                                                    ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Version: 6.0.0 (Production)
"""

import os
import io
import re
import json
import zipfile
import asyncio
from typing import Optional, Dict, List, Any, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

# Third-party imports
from dateutil.relativedelta import relativedelta
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

VERSION = "6.0.0"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Month name to number mapping
MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         SKILL CATEGORIES                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

SKILL_CATEGORIES = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "cobol", "bash", "shell",
        "powershell", "sql", "plsql", "t-sql", "r", "matlab", "perl", "groovy",
        "vb.net", "asp.net", "vbscript"
    ],
    "Web & Frameworks": [
        "html", "css", "react", "angular", "vue", "node.js", "nodejs", "express",
        "django", "flask", "fastapi", ".net", "asp.net", "spring", "spring boot",
        "hibernate", "struts", "ejb", "jquery", "bootstrap", "tailwind", "spring batch",
        "spring mvc", "spring jpa", "webflux", "micronaut"
    ],
    "Cloud Platforms": [
        "aws", "azure", "gcp", "google cloud", "ibm cloud", "openstack",
        "ec2", "s3", "lambda", "ecs", "ecr", "eks", "rds", "vpc", "route 53",
        "cloudformation", "cloudwatch", "sns", "sqs", "dynamodb", "aurora",
        "azure devops", "azure sql", "bigquery", "dataproc", "dataflow", "composer"
    ],
    "Data Engineering": [
        "etl", "data warehouse", "data pipeline", "informatica", "ssis", "ssas", "ssrs",
        "talend", "apache airflow", "airflow", "apache spark", "spark", "pyspark",
        "apache kafka", "kafka", "snowflake", "databricks", "dbt", "glue", "redshift",
        "data lake", "bigquery", "nifi", "hdfs", "oozie", "sqoop", "hue"
    ],
    "Databases": [
        "mongodb", "cassandra", "redis", "postgresql", "mysql", "oracle", "db2",
        "sql server", "nosql", "dynamodb", "elasticsearch", "neo4j", "teradata",
        "vertica", "singlestore", "hive", "hbase", "couchdb", "mariadb", "postgres"
    ],
    "DevOps & CI/CD": [
        "docker", "kubernetes", "k8s", "jenkins", "gitlab", "github actions",
        "ci/cd", "cicd", "concourse", "helm", "gitops", "devops", "devsecops",
        "azure devops", "octopus", "bamboo", "travis ci", "circleci", "argo",
        "maven", "gradle", "ant"
    ],
    "Infrastructure & Virtualization": [
        "terraform", "ansible", "puppet", "chef", "vagrant", "packer",
        "vmware", "vmware vsphere", "vcloud", "virtualization", "hyper-v",
        "openshift", "rancher", "istio", "consul", "vault", "cloudformation"
    ],
    "Testing & QA": [
        "selenium", "pytest", "unittest", "testng", "junit", "cucumber", "behave",
        "bdd", "tdd", "manual testing", "automation testing", "regression testing",
        "api testing", "postman", "functional testing", "uat", "quality assurance",
        "jmeter", "sonarqube", "loadrunner", "cypress", "playwright", "mockito"
    ],
    "Security": [
        "qualys", "crowdstrike", "mend", "uptycs", "snyk", "veracode",
        "siem", "splunk", "palo alto", "firewall", "iam", "pam", "mfa",
        "cisco security", "checkpoint", "fortinet", "zscaler", "oauth", "jwt"
    ],
    "Business Intelligence & Visualization": [
        "tableau", "power bi", "grafana", "prometheus", "elk stack", "kibana",
        "ssrs", "ssas", "obiee", "looker", "qlik", "spotfire", "datadog",
        "elastic search", "logstash", "business objects"
    ],
    "Project Management & Collaboration": [
        "jira", "confluence", "servicenow", "agile", "scrum", "kanban",
        "waterfall", "pmp", "prince2", "itil", "ms project", "clarity-ppm",
        "monday.com", "asana", "trello", "azure boards", "smartsheet"
    ],
    "Messaging & Integration": [
        "kafka", "rabbitmq", "ibm mq", "activemq", "sqs", "sns", "pubsub",
        "rest", "soap", "graphql", "grpc", "api gateway", "mulesoft", "apigee",
        "ibm message broker"
    ],
    "Version Control": [
        "git", "github", "gitlab", "bitbucket", "svn", "mercurial", "perforce",
        "tfs", "vsts"
    ],
    "Operating Systems": [
        "linux", "ubuntu", "rhel", "centos", "debian", "unix", "windows server",
        "macos", "redhat", "fedora", "alpine", "aix", "solaris"
    ],
    "Business Domains": [
        "banking", "finance", "financial services", "insurance", "healthcare",
        "telecom", "retail", "e-commerce", "manufacturing", "pharma",
        "investment banking", "capital markets", "payments", "billing"
    ]
}

# Flatten all skills for quick lookup
ALL_SKILLS: Set[str] = set()
for skills in SKILL_CATEGORIES.values():
    ALL_SKILLS.update(s.lower() for s in skills)

# Location patterns
LOCATION_CITIES = [
    'Philadelphia', 'Chicago', 'Dallas', 'Plano', 'Houston', 'Atlanta', 
    'Bangalore', 'Bengaluru', 'Hyderabad', 'North Chicago', 'New York',
    'San Francisco', 'Des Moines', 'Seattle', 'Boston', 'Denver', 'Austin',
    'Los Angeles', 'Portland', 'Charlotte', 'Raleigh', 'Nashville', 'Tampa',
    'Noida', 'Gurgaon', 'Pune', 'Mumbai', 'Chennai', 'Kolkata', 'France',
    'North America', 'India', 'USA', 'UK', 'Germany', 'Singapore', 'Canada'
]
LOCATION_PATTERN = '|'.join(re.escape(city) for city in LOCATION_CITIES[:35])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           DATA CLASSES                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class ValidationResult:
    """Result from Validation Agent."""
    score: int = 100
    issues: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    needs_ai_enhancement: bool = False


class ParseTextRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Resume text to parse")
    use_ai_validation: bool = Field(default=True, description="Enable AI enhancement")
    filename: Optional[str] = Field(default=None, description="Original filename")


class ExtractSkillsRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to extract skills from")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          FASTAPI APPLICATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

app = FastAPI(
    title="Resume Parser API",
    description=f"""
## Enterprise Resume Parser v{VERSION} - Complete Agentic Framework

### 🏗️ Architecture
Multi-agent system with specialized components:

| Agent | Purpose | Details |
|-------|---------|---------|
| **Extraction Agent** | Pattern-based parsing | 7 regex patterns for various formats |
| **Validation Agent** | Quality assurance | Scoring (0-100), issue detection |
| **AI Enhancement Agent** | Gap filling | Claude API for missing fields |
| **Output Agent** | Standardization | Clean JSON generation |

### 📄 Supported Resume Patterns
1. **Standard**: "Company – Title Date" (most common)
2. **Worked As**: "Worked as X in Y from A to B"
3. **Table**: "Client: X | Duration: Y"
4. **Title-Date-Client**: "Title (Date) – Client – Employer"
5. **Two-Line**: "Company Date" then "Title" next line
6. **Pipe**: "Title | Date" with company above
7. **Keyword**: "ROLE:" / "DESIGNATION:" format

### 📁 Supported File Formats
- **PDF**: Standard and multi-column layouts
- **DOCX**: Tables, text boxes, complex formatting
- **TXT**: Plain text
- **ZIP**: Archives containing text files

### 🎯 Features
- 15+ skill categories with experience months calculation
- Intelligent file type detection
- Deduplication and data quality checks
- Comprehensive error handling
    """,
    version=VERSION,
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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       FILE EXTRACTION UTILITIES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def detect_file_type(content: bytes, filename: str) -> str:
    """Detect actual file type from magic bytes."""
    if content[:4] == b'PK\x03\x04':
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = zf.namelist()
                if any('word/document.xml' in n for n in names):
                    return 'docx'
                elif any(n.endswith('.txt') for n in names):
                    return 'zip_text'
                return 'zip'
        except:
            pass
    elif content[:5] == b'%PDF-':
        return 'pdf'
    elif content[:4] == b'\xd0\xcf\x11\xe0':
        return 'doc'
    
    # Check if it's actually text content (common for markdown/text files with wrong extension)
    try:
        # Try to decode as text - if it works and looks like text, treat as text
        text_sample = content[:1000].decode('utf-8', errors='strict')
        # Check for common text/markdown indicators
        if text_sample.startswith(('#', '##', '**', 'PROFESSIONAL', 'SUMMARY', 'RESUME', '-')):
            return 'txt'
        # Check if mostly printable ASCII
        printable_ratio = sum(1 for c in text_sample if c.isprintable() or c in '\n\r\t') / len(text_sample)
        if printable_ratio > 0.9:
            return 'txt'
    except:
        pass
    
    ext = filename.lower().split('.')[-1] if filename else ''
    return ext if ext in ['pdf', 'docx', 'doc', 'txt'] else 'unknown'


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF using multiple methods."""
    text = ""
    
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception:
        pass
    
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        pass
    
    return text


def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX including tables and text boxes."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(content))
        
        all_text = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                all_text.append(text)
        
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_text.append(cell_text)
                if row_text:
                    all_text.append(' | '.join(row_text))
        
        try:
            xml_str = doc.element.xml
            pattern = r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>'
            matches = re.findall(pattern, xml_str, re.DOTALL)
            for match in matches:
                text_pattern = r'<w:t[^>]*>([^<]+)</w:t>'
                texts = re.findall(text_pattern, match)
                if texts:
                    combined = ' '.join(texts)
                    combined = ' '.join(combined.split())
                    if combined and len(combined) > 5:
                        all_text.append(combined)
        except:
            pass
        
        return '\n'.join(all_text)
    except Exception:
        return ""


def extract_text_from_zip(content: bytes) -> str:
    """Extract text from ZIP archive."""
    text_parts = []
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith('.txt'):
                    text_parts.append(zf.read(name).decode('utf-8', errors='ignore'))
    except:
        pass
    return '\n'.join(text_parts)


def extract_text_intelligent(content: bytes, filename: str) -> str:
    """Intelligently extract text based on actual file type."""
    file_type = detect_file_type(content, filename)
    
    if file_type == 'pdf':
        return extract_text_from_pdf(content)
    elif file_type in ['docx', 'doc']:
        return extract_text_from_docx(content)
    elif file_type == 'zip_text':
        return extract_text_from_zip(content)
    elif file_type == 'zip':
        text = extract_text_from_zip(content)
        if not text.strip():
            text = extract_text_from_docx(content)
        return text
    else:
        try:
            return content.decode('utf-8', errors='ignore')
        except:
            return ""


def normalize_text(text: str) -> str:
    """Clean and normalize resume text."""
    if not text:
        return ""
    
    replacements = {
        'â€"': '–', 'â€™': "'", 'â€œ': '"', 'â€': '"',
        'Ã©': 'é', 'Ã¨': 'è', 'Ã ': 'à',
        '\u2013': '-', '\u2014': '-', '–': '-',
        '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2022': '•', '\u00a0': ' ',
        '\r\n': '\n', '\r': '\n', '\t': ' '
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = re.sub(r'Pres\s*ent', 'Present', text, flags=re.IGNORECASE)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        DATE PARSING UTILITIES                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def parse_date(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse date string to (year, month, is_present)."""
    if not text:
        return None, None, False
    
    text = text.strip().lower()
    
    if any(p in text for p in ['present', 'current', 'now', 'ongoing', 'till date']):
        now = datetime.now()
        return now.year, now.month, True
    
    # Month Year format (full year)
    match = re.search(
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
        r'aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
        r'\s*[,.]?\s*(\d{4})', text
    )
    if match:
        month = MONTH_MAP.get(match.group(1)[:3])
        year = int(match.group(2))
        return year, month, False
    
    # Compact with optional apostrophe: Jan2021, Jul24, Feb'20, May'14
    match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)['\s]?(\d{2,4})", text, re.IGNORECASE)
    if match:
        month = MONTH_MAP.get(match.group(1).lower())
        year_str = match.group(2)
        year = 2000 + int(year_str) if len(year_str) == 2 and int(year_str) < 50 else (
            1900 + int(year_str) if len(year_str) == 2 else int(year_str)
        )
        return year, month, False
    
    # Month with space and 2-digit year: "July 15"
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|'
                      r'aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
                      r'\s+(\d{2})(?:\s|$)', text)
    if match:
        month = MONTH_MAP.get(match.group(1)[:3])
        year_str = match.group(2)
        year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        return year, month, False
    
    # Just year
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if match:
        return int(match.group(1)), None, False
    
    return None, None, False


def calculate_duration(start_year: int, start_month: int, end_year: int, end_month: int) -> int:
    """Calculate duration in months (inclusive)."""
    start_dt = datetime(start_year, start_month or 1, 1)
    end_dt = datetime(end_year, end_month or 12, 1)
    delta = relativedelta(end_dt, start_dt)
    return max(1, delta.years * 12 + delta.months + 1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     EXTRACTION AGENT - CONTACT INFO                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_contact(text: str) -> Dict[str, str]:
    """Extract contact information from resume text."""
    contact = {'email': '', 'phone': '', 'linkedin': '', 'location': ''}
    
    # Email
    match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text)
    if match:
        contact['email'] = match.group(1).lower()
    
    # Phone
    phone_patterns = [
        r'(?:Mob|Phone|Tel|Mobile|Cell|Ph)[:\s]*(\+?[\d\s\-().]{10,})',
        r'(\+1\s*\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
        r'(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
        r'(\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
        r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phone = re.sub(r'[^\d+\-() ]', '', match.group(1)).strip()
            if len(re.sub(r'\D', '', phone)) >= 10:
                contact['phone'] = phone
                break
    
    # LinkedIn
    linkedin_patterns = [
        r'linkedin\.com/in/([\w-]+)',
        r'LinkedIn[:\s]+([\w-]+)',
    ]
    for pattern in linkedin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            username = match.group(1)
            if username.lower() not in ['summary', 'profile', 'in', 'phone', 'email']:
                contact['linkedin'] = f"www.linkedin.com/in/{username}"
                break
    
    # Location
    for city in LOCATION_CITIES[:30]:
        pattern = rf'\b{re.escape(city)}[,\s]+(PA|TX|IL|NY|CA|GA|OH|IN|India|USA|Karnataka|Maharashtra|UK|Germany)\b'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            contact['location'] = f"{city}, {match.group(1)}"
            break
    
    return contact


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     EXTRACTION AGENT - NAME EXTRACTION                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_name(text: str) -> Tuple[str, str, str]:
    """Extract first, middle, last name from resume."""
    lines = [l.strip() for l in text.split('\n') if l.strip()][:80]  # Extend to 80 lines for two-column layouts
    
    skip_patterns = [
        'resume', 'cv', 'curriculum vitae', 'key expertise', 'professional experience',
        'education', 'skills', 'summary', 'objective', 'technical', 'profile',
        'results-driven', 'experienced', 'skilled', 'dedicated', 'total',
        'data engineering', 'core competencies', 'with over', 'contact', 'houston',
        'years of experience', 'developer with', 'professional summary', 'spring',
        'java', 'python', 'aws', 'gcp', 'sql', 'micro', 'kafka', 'rest', 'xml',
        'docker', 'kubernetes', 'jenkins', 'git', 'linux', 'cloud', 'database',
        'web flux', 'hibernate', 'junit', 'jboss', 'tomcat', 'maven', 'gradle'
    ]
    
    # Tech terms that look like names but aren't
    tech_terms = [
        'SQL', 'ETL', 'GCP', 'AWS', 'API', 'XML', 'JSON', 'HTML', 'CSS', 'SSIS',
        'Spring', 'JPA', 'Java', 'Hibernate', 'EJB', 'REST', 'SOAP', 'Kafka',
        'Docker', 'Maven', 'Gradle', 'Jenkins', 'Git', 'Linux', 'Unix', 'Python',
        'React', 'Angular', 'Node', 'Vue', 'MongoDB', 'Redis', 'MySQL', 'Oracle',
        'Spark', 'Hadoop', 'Hive', 'Scala', 'Kibana', 'Elastic', 'Grafana', 'Prometheus'
    ]
    
    # Look for name in first pass - typical positions
    for line in lines:
        clean_line = ' '.join(line.split())
        
        # Skip lines with bullet points or special characters indicating skills
        if '■' in clean_line or '•' in clean_line or '|' in clean_line:
            continue
        # Skip lines that contain commas (likely skill lists)
        if ',' in clean_line and len(clean_line.split(',')) > 2:
            continue
        
        # Remove markdown formatting
        clean_line = re.sub(r'^#+\s*', '', clean_line)  # Remove ## headers
        clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_line)  # Remove **bold**
        clean_line = re.sub(r'>\s*', '', clean_line)  # Remove > quotes
        clean_line = re.sub(r'\[([^\]]+)\]\{[^}]*\}', r'\1', clean_line)  # Remove [text]{.class}
        
        # Remove contact info (including "Contact: phone" pattern)
        name_part = re.sub(
            r'\s*(Contact|Phone|Email|Tel|Cell|Mobile)[:\s].*$', 
            '', clean_line, flags=re.IGNORECASE
        ).strip()
        
        if any(name_part.lower().startswith(skip) for skip in skip_patterns):
            continue
        # Also skip if any skip pattern is contained in the line
        if any(skip in name_part.lower() for skip in ['spring', 'java', 'python', 'sql', 'aws', 'hibernate', 'kafka', 'docker']):
            continue
        if name_part.endswith(':'):
            continue
        if re.match(r'^[\w.+-]+@[\w.-]+\.\w+$', name_part):
            continue
        if re.match(r'^[\d\s\-+()]+$', name_part):
            continue
        # Skip lines that are clearly not names
        if len(name_part) > 60:
            continue
        # Skip lines with common resume headers
        if name_part.upper() in ['EXPERIENCE', 'PROFESSIONAL SUMMARY', 'SKILLS', 'CONTACT']:
            continue
        
        name = re.sub(r'[\|].*$', '', name_part).strip()
        name = re.sub(r'\s*,.*$', '', name).strip()
        
        parts = name.split()
        
        if 2 <= len(parts) <= 4:
            if all(p[0].isupper() for p in parts if p):
                # Check if any part is a tech term
                if any(p in tech_terms for p in parts):
                    continue
                if len(parts) == 2:
                    return parts[0], "", parts[1]
                elif len(parts) == 3:
                    return parts[0], parts[1], parts[2]
                else:
                    return parts[0], ' '.join(parts[1:-1]), parts[-1]
    
    return "", "", ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              EXTRACTION AGENT - EXPERIENCE (7 PATTERNS)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_experiences(text: str) -> List[Dict]:
    """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                    EXTRACTION AGENT - 7 PATTERNS                            ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║ Pattern 1: Standard "Company – Title Date" format                           ║
    ║ Pattern 2: "Worked as X in Y from A to B" format                            ║
    ║ Pattern 3: Table format "Client: X | Duration: Y"                           ║
    ║ Pattern 4: "Title (DateRange) – Client – Employer"                          ║
    ║ Pattern 5: "Company Date" then "Title Location" next line                   ║
    ║ Pattern 6: Pipe format "Title | Date" with company above                    ║
    ║ Pattern 7: "ROLE:" / "DESIGNATION:" keyword format                          ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    experiences = []
    text = normalize_text(text)
    lines = text.split('\n')
    
    # Date range pattern
    date_pattern = (
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
        r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})'
        r'\s*[-–to]+\s*'
        r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
        r'Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}'
        r'|Present|Current|Till\s+[Dd]ate)'
    )
    
    # Title keywords for classification
    title_keywords = ['Engineer', 'Developer', 'Manager', 'Analyst', 'Lead', 
                      'Consultant', 'Architect', 'Specialist', 'Director',
                      'Admin', 'Coordinator', 'Scrum', 'Master', 'Tester',
                      'Designer', 'Administrator', 'Executive', 'Officer']
    
    # =========================================================================
    # PATTERN 4: "Title (DateRange) – Client – Employer" (Nageswara)
    # Example: GCP Data Engineer (Jan2021 - Present) – Renault-Nissan – Atos Global IT Solution
    # Also handles markdown: > **GCP Data Engineer (Jan2021 - Present) -- [Renault-Nissan -- Atos
    # =========================================================================
    title_date_client_pattern = (
        r'^\s*>?\s*\*?\*?\s*'  # Optional > and markdown
        r'([\w\s/]+?)'  # Title (e.g., "GCP Data Engineer")
        r'\s*\((\w{3,9}\d{2,4})\s*[-–]+\s*(\w{3,9}\d{2,4}|Present|Current)\)'  # (Date - Date)
        r'\s*[-–]+\s*\[?'  # – or -- with optional [
        r'([^-–\[\]]+?)'  # Client (e.g., "Renault-Nissan")
        r'\s*[-–]+\s*'  # – or --
        r'\*?\*?\[?\*?\*?'  # Optional **[**
        r'(.+)$'  # Employer (rest of line - greedy)
    )
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        match = re.match(title_date_client_pattern, clean_line, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            start_str = match.group(2)
            end_str = match.group(3)
            client = match.group(4).strip().strip('[').strip(']').strip('*').strip()
            employer = match.group(5).strip().strip(']').strip('[').strip('*').strip()
            
            # Clean up employer (remove trailing underlines or garbage)
            employer = re.sub(r'\s*\[.*$', '', employer)
            employer = employer.rstrip('_').strip()
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                # Extract responsibilities from table format
                responsibilities = []
                for j in range(i + 1, min(len(lines), i + 40)):
                    resp_line = lines[j].strip()
                    # Stop at next experience header
                    if re.match(r'^\s*>?\s*\*?\*?\s*[\w\s/]+\s*\(\w{3,9}\d{2,4}', resp_line, re.IGNORECASE):
                        break
                    # Extract from table cells: "| - text |" or "| text |"
                    if '|' in resp_line:
                        resp_match = re.findall(r'\|\s*[-•]?\s*([^|]+)', resp_line)
                        for resp in resp_match:
                            resp = resp.strip().strip('-').strip('*').strip()
                            if len(resp) > 25 and not resp.lower().startswith('environment'):
                                responsibilities.append(resp[:500])
                
                is_dup = any(
                    e.get('start_date') == start_date and
                    (e.get('title', '').lower() == title.lower())
                    for e in experiences
                )
                
                if not is_dup:
                    experiences.append({
                        'employer': f"{employer} (Client: {client})" if client != employer else employer,
                        'title': title,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_months': duration,
                        'responsibilities': responsibilities[:12],
                        'location': None,
                        'client': client,
                        'pattern': 4
                    })
    
    # =========================================================================
    # PATTERN 9: "Title Company - Location" then "MM/YYYY - Present" (Javvaji)
    # Example: Senior Java Developer LOWE'S - USA
    #          05/2023 - Present
    # =========================================================================
    mm_yyyy_pattern = r'^(\d{1,2}/\d{4})\s*[-–]\s*(\d{1,2}/\d{4}|Present|Current)$'
    
    for i, line in enumerate(lines):
        clean_line = line.strip()
        date_match = re.match(mm_yyyy_pattern, clean_line, re.IGNORECASE)
        
        if date_match and i > 0:
            start_str = date_match.group(1)
            end_str = date_match.group(2)
            
            # Parse MM/YYYY format
            start_parts = start_str.split('/')
            if len(start_parts) == 2:
                start_month = int(start_parts[0])
                start_year = int(start_parts[1])
            else:
                continue
            
            if 'present' in end_str.lower() or 'current' in end_str.lower():
                end_year = datetime.now().year
                end_month = datetime.now().month
                is_present = True
            else:
                end_parts = end_str.split('/')
                if len(end_parts) == 2:
                    end_month = int(end_parts[0])
                    end_year = int(end_parts[1])
                    is_present = False
                else:
                    continue
            
            # Previous line should be "Title Company - Location"
            prev_line = lines[i - 1].strip()
            if not prev_line or prev_line.startswith(('•', '-', '*', '■')):
                continue
            
            # Parse "Title Company - Location" format
            # E.g., "Senior Java Developer LOWE'S - USA"
            title, employer, location = "", "", ""
            
            # Remove location suffix first
            loc_match = re.search(r'\s*[-–]\s*(USA|France|North America|India|UK|Germany|[A-Z][a-z]+(?:\s*&\s*[A-Z][a-z]+)?)$', prev_line, re.IGNORECASE)
            if loc_match:
                location = loc_match.group(1).strip()
                title_company = prev_line[:loc_match.start()].strip()
            else:
                title_company = prev_line
            
            # Try to split "Title Company" - look for pattern where title ends
            # Pattern: title keywords followed by company name
            # E.g., "Senior Java Developer LOWE'S" -> Title = "Senior Java Developer", Employer = "LOWE'S"
            # E.g., "Java Consultant Renault-Nissan" -> Title = "Java Consultant", Employer = "Renault-Nissan"
            
            # Find where title likely ends (after last title keyword)
            title_end_idx = 0
            for kw in title_keywords:
                kw_lower = kw.lower()
                tc_lower = title_company.lower()
                if kw_lower in tc_lower:
                    idx = tc_lower.find(kw_lower) + len(kw)
                    if idx > title_end_idx:
                        title_end_idx = idx
            
            if title_end_idx > 0:
                title = title_company[:title_end_idx].strip()
                employer = title_company[title_end_idx:].strip()
                # Remove leading spaces or dashes
                employer = re.sub(r'^[\s\-–]+', '', employer).strip()
            else:
                # Fallback: just use the whole thing as title
                title = title_company
                employer = None
            
            if start_year and end_year:
                duration = calculate_duration(start_year, start_month, end_year, end_month)
                start_date = f"{start_year}-{start_month:02d}"
                end_date = f"{end_year}-{end_month:02d}"
                
                # Extract responsibilities (lines with ■ bullets)
                responsibilities = []
                for j in range(i + 1, min(len(lines), i + 40)):
                    resp_line = lines[j].strip()
                    # Stop at next date pattern
                    if re.match(mm_yyyy_pattern, resp_line, re.IGNORECASE):
                        break
                    if resp_line.startswith(('■', '•', '-', '*')) or re.match(r'^\d+\.', resp_line):
                        resp = re.sub(r'^[■•\-\*\d.]+\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(resp[:500])
                
                is_dup = any(
                    e.get('start_date') == start_date and
                    e.get('title', '').lower()[:20] == title.lower()[:20]
                    for e in experiences
                )
                
                if not is_dup and title:
                    experiences.append({
                        'employer': employer if employer else None,
                        'title': title,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_months': duration,
                        'responsibilities': responsibilities[:12],
                        'location': location or None,
                        'pattern': 9
                    })
    
    # =========================================================================
    # PATTERN 7: "ROLE:" / "DESIGNATION:" keyword format (Sarwer)
    # Process this FIRST to capture structured formats
    # =========================================================================
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for company headers followed by ROLE/DESIGNATION
        if line and not line.startswith(('•', '-', '*', '■')):
            # Check for date pattern on same or nearby line
            date_match = None
            for check_idx in range(max(0, i-1), min(len(lines), i+3)):
                check_line = lines[check_idx].strip()
                dm = re.search(date_pattern, check_line, re.IGNORECASE)
                if dm:
                    date_match = dm
                    break
            
            if date_match:
                # Look for ROLE: or DESIGNATION: nearby
                for role_idx in range(i, min(len(lines), i + 8)):
                    role_line = lines[role_idx].strip()
                    role_match = re.match(r'^(?:ROLE|DESIGNATION)\s*[:\s]+\s*(.+)$', role_line, re.IGNORECASE)
                    
                    if role_match:
                        title = role_match.group(1).strip()
                        
                        # Find employer (should be on the date line or above it)
                        employer = None
                        for emp_idx in range(role_idx - 1, max(0, role_idx - 6), -1):
                            emp_line = lines[emp_idx].strip()
                            if emp_line and not emp_line.startswith(('•', '-', '*', '■', 'ROLE', 'DESIGNATION')):
                                # Skip "Work Location" lines - these are NOT employers
                                if 'work location' in emp_line.lower():
                                    continue
                                # Skip pure date lines
                                if re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', emp_line, re.IGNORECASE):
                                    continue
                                if len(emp_line) > 3 and len(emp_line) < 100:
                                    # Skip section headers
                                    if emp_line.upper() not in ['WORK EXPERIENCE', 'EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'ROLES & RESPONSIBILITIES']:
                                        # If line has date pattern, extract company (text before date)
                                        dm = re.search(date_pattern, emp_line, re.IGNORECASE)
                                        if dm:
                                            employer = emp_line[:dm.start()].strip().rstrip('-–')
                                        else:
                                            employer = emp_line
                                        if employer and len(employer) > 2:
                                            break
                        
                        if employer and title:
                            start_str, end_str = date_match.group(1), date_match.group(2)
                            start_year, start_month, _ = parse_date(start_str)
                            end_year, end_month, is_present = parse_date(end_str)
                            
                            if start_year and end_year:
                                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                                start_date = f"{start_year}-{(start_month or 1):02d}"
                                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                                
                                # Extract responsibilities
                                responsibilities = []
                                for resp_idx in range(role_idx + 1, min(len(lines), role_idx + 20)):
                                    resp_line = lines[resp_idx].strip()
                                    if re.match(r'^(?:ROLE|DESIGNATION|[A-Z][a-z]+\s+\d{4})', resp_line):
                                        break
                                    if re.search(date_pattern, resp_line, re.IGNORECASE):
                                        break
                                    if resp_line.startswith(('•', '-', '*', '■')) or re.match(r'^\d+\.', resp_line):
                                        resp = re.sub(r'^[•\-\*■\d.]+\s*', '', resp_line)
                                        if len(resp) > 20:
                                            responsibilities.append(resp[:500])
                                
                                # Check for duplicate
                                is_dup = any(
                                    e.get('start_date') == start_date and
                                    (e.get('employer', '').lower() == employer.lower() or
                                     e.get('title', '').lower() == title.lower())
                                    for e in experiences
                                )
                                
                                if not is_dup:
                                    experiences.append({
                                        'employer': employer,
                                        'title': title,
                                        'start_date': start_date,
                                        'end_date': end_date,
                                        'duration_months': duration,
                                        'responsibilities': responsibilities[:12],
                                        'location': None,
                                        'pattern': 7
                                    })
                        break
        i += 1
    
    # =========================================================================
    # PATTERN 8: "**Client: Company -- Location Date to Date**" (Naveen)
    # Handles hyphenated names like "Co-Op Financial" by requiring space before separator
    # =========================================================================
    # Match " - " or " – " (with spaces) as separator to avoid splitting hyphenated names
    client_pattern = r'\*?\*?Client:\s*(.+?)\s+[-–]\s+([A-Za-z][^.]+?)\.?\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\']*\s*\d{2,4})\s+to\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\']*\s*\d{2,4}|Present|Current|July\s*\d{2,4})\*?\*?'
    
    for i, line in enumerate(lines):
        clean_line = re.sub(r'\*\*', '', line.strip())
        match = re.match(client_pattern, clean_line, re.IGNORECASE)
        
        if match:
            employer = match.group(1).strip()
            location = match.group(2).strip()
            start_str = match.group(3)
            end_str = match.group(4)
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                # Look for title on next line
                title = ""
                if i + 1 < len(lines):
                    next_line = re.sub(r'\*\*', '', lines[i + 1].strip())
                    if next_line and not next_line.startswith(('Client:', '•', '-', '*', '■')):
                        if not re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line, re.IGNORECASE):
                            title = next_line
                
                # Extract responsibilities
                responsibilities = []
                for j in range(i + 2, min(len(lines), i + 25)):
                    resp_line = lines[j].strip()
                    resp_line = re.sub(r'\*\*', '', resp_line)
                    
                    if re.match(r'\*?\*?Client:', resp_line, re.IGNORECASE):
                        break
                    if resp_line.startswith(('•', '-', '*', '■')) or re.match(r'^\d+\.', resp_line):
                        resp = re.sub(r'^[•\-\*■\d.]+\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(resp[:500])
                
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                is_dup = any(
                    e.get('start_date') == start_date and
                    e.get('employer', '').lower() == employer.lower()
                    for e in experiences
                )
                
                if not is_dup:
                    experiences.append({
                        'employer': employer,
                        'title': title or None,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_months': duration,
                        'responsibilities': responsibilities[:12],
                        'location': location,
                        'pattern': 8
                    })
    
    # =========================================================================
    # PATTERN 8b: "Client: Company, Location Date to Date" (Naveen comma variant)
    # Format: Client: UnitedHealth Group, India May'14 to Dec'14
    # =========================================================================
    client_pattern_b = r'\*?\*?Client:\s*(.+?),\s*([A-Za-z][A-Za-z\s]+)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\']*\s*\d{2,4})\s+to\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z\']*\s*\d{2,4}|Present|Current)\*?\*?'
    
    for i, line in enumerate(lines):
        clean_line = re.sub(r'\*\*', '', line.strip())
        match = re.match(client_pattern_b, clean_line, re.IGNORECASE)
        
        if match:
            employer = match.group(1).strip()
            location = match.group(2).strip()
            start_str = match.group(3)
            end_str = match.group(4)
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                # Look for title on next line
                title = ""
                if i + 1 < len(lines):
                    next_line = re.sub(r'\*\*', '', lines[i + 1].strip())
                    if next_line and not next_line.startswith(('Client:', '•', '-', '*', '■')):
                        if not re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', next_line, re.IGNORECASE):
                            title = next_line
                
                # Extract responsibilities
                responsibilities = []
                for j in range(i + 2, min(len(lines), i + 25)):
                    resp_line = lines[j].strip()
                    resp_line = re.sub(r'\*\*', '', resp_line)
                    
                    if re.match(r'\*?\*?Client:', resp_line, re.IGNORECASE):
                        break
                    if resp_line.startswith(('•', '-', '*', '■')) or re.match(r'^\d+\.', resp_line):
                        resp = re.sub(r'^[•\-\*■\d.]+\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(resp[:500])
                
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                is_dup = any(
                    e.get('start_date') == start_date and
                    e.get('employer', '').lower() == employer.lower()
                    for e in experiences
                )
                
                if not is_dup:
                    experiences.append({
                        'employer': employer,
                        'title': title or None,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_months': duration,
                        'responsibilities': responsibilities[:12],
                        'location': location,
                        'pattern': 8
                    })
    
    # =========================================================================
    # PATTERN 10: "## Title | Company, Location | Date" (Steven markdown)
    # Format: ## Sr.Cloud/DeVOPS Engineer | RElix, Tx | November 2022 -- Present
    # =========================================================================
    steven_pattern = r'^\s*##\s*(.+?)\s*\|\s*([^|,]+)(?:,\s*([^|]+))?\s*\|\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–]+\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current)'
    
    for i, line in enumerate(lines):
        match = re.match(steven_pattern, line.strip(), re.IGNORECASE)
        
        if match:
            title = match.group(1).strip()
            employer = match.group(2).strip()
            location = match.group(3).strip() if match.group(3) else None
            start_str = match.group(4)
            end_str = match.group(5)
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                # Extract responsibilities
                responsibilities = []
                for j in range(i + 1, min(len(lines), i + 40)):
                    resp_line = lines[j].strip()
                    
                    # Stop at next experience
                    if re.match(r'^\s*##\s*.+\|', resp_line):
                        break
                    if re.match(r'^#\s+\w+', resp_line):  # New section
                        break
                    
                    if resp_line.startswith(('-', '•', '*')) or re.match(r'^\d+\.', resp_line):
                        resp = re.sub(r'^[-•*\d.]+\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(resp[:500])
                
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                is_dup = any(
                    e.get('start_date') == start_date and
                    e.get('employer', '').lower() == employer.lower()
                    for e in experiences
                )
                
                if not is_dup:
                    experiences.append({
                        'employer': employer,
                        'title': title,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_months': duration,
                        'responsibilities': responsibilities[:12],
                        'location': location,
                        'pattern': 10
                    })
    
    # =========================================================================
    # PATTERN 2: "Worked as X in Y from A to B" (Madhuri)
    # =========================================================================
    normalized_text = re.sub(r'\*\*', '', text)
    normalized_text = re.sub(r'\n\s*>\s*', ' ', normalized_text)
    normalized_text = re.sub(r'\s+', ' ', normalized_text)
    
    worked_pattern = (
        r'[Ww]ork(?:ed|ing)\s+(?:as\s+)?(?:a\s+)?(.+?)\s+in\s+(.+?)\s+from\s+'
        r'(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|Present|Current)'
    )
    
    for match in re.finditer(worked_pattern, normalized_text, re.IGNORECASE):
        title = match.group(1).strip()
        employer = match.group(2).strip()
        start_year, start_month, _ = parse_date(match.group(3))
        end_year, end_month, is_present = parse_date(match.group(4))
        
        if start_year and end_year:
            duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
            start_date = f"{start_year}-{(start_month or 1):02d}"
            end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
            
            is_dup = any(
                e.get('start_date') == start_date and
                (e.get('employer', '').lower() in employer.lower() or employer.lower() in e.get('employer', '').lower())
                for e in experiences
            )
            
            if not is_dup:
                experiences.append({
                    'employer': employer,
                    'title': title,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_months': duration,
                    'responsibilities': [],
                    'location': None,
                    'pattern': 2
                })
    
    # =========================================================================
    # PATTERNS 1, 4, 5, 6: Standard date-based extraction
    # =========================================================================
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        line_stripped = re.sub(r'^[>\-•*#]+\s*', '', line).strip()
        
        if not line_stripped or line_stripped.startswith(('•', '-', '*', '■')):
            i += 1
            continue
        
        date_match = re.search(date_pattern, line_stripped, re.IGNORECASE)
        
        if date_match:
            # Skip "Worked as" lines (handled by Pattern 2)
            if re.search(r'\bwork(?:ed|ing)\s+(?:as\s+)?(?:a\s+)?', line_stripped, re.IGNORECASE):
                i += 1
                continue
            
            # Skip education lines
            edu_keywords = ['university', 'college', 'bachelor', 'master', 'mba', 'education', 'degree']
            if any(kw in line.lower() for kw in edu_keywords):
                i += 1
                continue
            
            # Skip section headers
            header_preview = line_stripped[:date_match.start()].strip().lower()
            if header_preview in ['work experience', 'experience', 'employment', 'professional experience']:
                i += 1
                continue
            
            start_str, end_str = date_match.group(1), date_match.group(2)
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                employer, title = "", ""
                header = line_stripped[:date_match.start()].strip()
                
                # PATTERN 4: "Title (DateRange) – Client – Employer"
                title_date_pattern = r'^([\w\s/]+)\s*\([^)]+\)\s*[-–]+\s*([^-–]+)[-–]+(.+)$'
                tdm = re.match(title_date_pattern, line_stripped, re.IGNORECASE)
                if tdm:
                    title = tdm.group(1).strip()
                    client = tdm.group(2).strip()
                    employer = tdm.group(3).strip()
                    if employer:
                        employer = f"{client} ({employer})"
                    else:
                        employer = client
                
                # PATTERN 6: Pipe format "Title | ... | Date"
                elif '|' in header:
                    parts = [p.strip() for p in header.split('|') if p.strip()]
                    if len(parts) >= 1:
                        title = parts[0].strip()
                    
                    if not employer:
                        for back in range(1, min(10, i + 1)):
                            prev = lines[i - back].strip()
                            prev = re.sub(r'^[>\-•*#]+\s*', '', prev).strip()
                            if not prev or prev.startswith(('•', '-', '*')):
                                continue
                            if re.search(date_pattern, prev):
                                continue
                            if prev.upper() in ['PROFESSIONAL EXPERIENCE', 'WORK EXPERIENCE', 'EXPERIENCE']:
                                continue
                            if '–' in prev or ' - ' in prev:
                                employer = re.split(r'\s*[-–]+\s*', prev)[0].strip()
                            else:
                                employer = prev
                            break
                
                # Standard "Company – Title" or "Title – Company"
                elif '–' in header or '--' in header or ' - ' in header:
                    parts = re.split(r'\s+[-–]+\s+|\s*--\s*', header, maxsplit=1)
                    if len(parts) == 2:
                        part1, part2 = parts[0].strip(), parts[1].strip()
                        
                        if any(kw in part1 for kw in title_keywords):
                            title, employer = part1, part2
                        elif any(kw in part2 for kw in title_keywords):
                            employer, title = part1, part2
                        else:
                            employer, title = part1, part2
                        
                        loc_pattern = rf'\s+({LOCATION_PATTERN})[,\s]*(PA|TX|IL|GA|NY|CA|OH|India|USA)?$'
                        if title:
                            title = re.sub(loc_pattern, '', title, flags=re.IGNORECASE).strip()
                        if employer:
                            employer = re.sub(loc_pattern, '', employer, flags=re.IGNORECASE).strip()
                    else:
                        employer = header
                
                # PATTERN 5: "Company Date" then "Title Location" next line
                else:
                    employer = header
                    
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        next_clean = re.sub(r'^[>\-•*#]+\s*', '', next_line).strip()
                        
                        if next_clean and not next_clean.startswith(('•', '-', '*', '■')):
                            if not re.search(date_pattern, next_clean, re.IGNORECASE):
                                if any(kw.lower() in next_clean.lower() for kw in title_keywords):
                                    loc_pattern = rf'\s+({LOCATION_PATTERN})[,\s]*(PA|TX|IL|GA|NY|CA|OH|India|USA)?$'
                                    title = re.sub(loc_pattern, '', next_clean, flags=re.IGNORECASE).strip()
                
                # Clean up employer
                if employer:
                    loc_pattern = rf'\s*[-–]+\s*({LOCATION_PATTERN})[,\s]*(PA|TX|IL|USA|India)?$'
                    employer = re.sub(loc_pattern, '', employer, flags=re.IGNORECASE).strip()
                    employer = re.sub(r'^Client[:\s]+', '', employer, flags=re.IGNORECASE).strip()
                
                # Extract responsibilities
                responsibilities = []
                j = i + 1
                while j < len(lines) and j < i + 30:
                    resp_line = lines[j].strip()
                    if re.search(date_pattern, resp_line, re.IGNORECASE):
                        break
                    if re.match(r'^(EDUCATION|TECHNICAL|SKILLS|CERTIFICATIONS|PERSONAL)', resp_line, re.IGNORECASE):
                        break
                    if resp_line.startswith(('•', '-', '*', '■', '>')) or re.match(r'^\d+\.', resp_line):
                        resp = re.sub(r'^[•\-\*■>\d.]+\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(resp[:500])
                    j += 1
                
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                is_dup = any(
                    e.get('start_date') == start_date and
                    (e.get('employer', '').lower() == (employer or '').lower() or
                     e.get('title', '').lower() == (title or '').lower())
                    for e in experiences
                )
                
                if not is_dup and (employer or title):
                    experiences.append({
                        'employer': employer.strip() if employer else None,
                        'title': title.strip() if title else None,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration_months': duration,
                        'responsibilities': responsibilities[:12],
                        'location': None,
                        'pattern': 1
                    })
        i += 1
    
    # =========================================================================
    # PATTERN 3: Table format "Client: X | Duration: Y" (Ramaswamy)
    # =========================================================================
    client_pattern = r'Client[:\s]+([^\|]+?)(?:\s*\||\s*Duration)'
    duration_pattern = r'Duration[:\s*]*(\w{3,9})[-/]?(\d{4})\s+to\s+(\w+)[-/\s]?(\d{4})?'
    
    for line in lines:
        clean_line = re.sub(r'\*+', '', line)
        client_match = re.search(client_pattern, clean_line, re.IGNORECASE)
        dur_match = re.search(duration_pattern, clean_line, re.IGNORECASE)
        
        if client_match and dur_match:
            employer = client_match.group(1).strip()
            start_month_str = dur_match.group(1)
            start_year = int(dur_match.group(2))
            end_str = dur_match.group(3).lower()
            end_year_str = dur_match.group(4)
            
            start_month = MONTH_MAP.get(start_month_str[:3].lower(), 1)
            
            if 'till' in end_str or 'present' in end_str or 'date' in end_str:
                end_year = datetime.now().year
                end_month = datetime.now().month
            else:
                end_month = MONTH_MAP.get(end_str[:3], 12)
                end_year = int(end_year_str) if end_year_str else start_year
            
            duration = calculate_duration(start_year, start_month, end_year, end_month)
            start_date = f"{start_year}-{start_month:02d}"
            end_date = f"{end_year}-{end_month:02d}"
            
            is_dup = any(
                e.get('start_date') == start_date and
                (e.get('employer', '').lower() in employer.lower() or employer.lower() in e.get('employer', '').lower())
                for e in experiences
            )
            
            if not is_dup:
                experiences.append({
                    'employer': employer,
                    'title': None,
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration_months': duration,
                    'responsibilities': [],
                    'location': None,
                    'pattern': 3
                })
    
    # Sort by start date (most recent first)
    experiences.sort(key=lambda x: x.get('start_date', ''), reverse=True)
    
    # Remove pattern field from output
    for exp in experiences:
        exp.pop('pattern', None)
    
    return experiences


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    EXTRACTION AGENT - EDUCATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_education(text: str) -> List[Dict]:
    """Extract education from resume text."""
    education = []
    seen_degrees = set()
    
    edu_match = re.search(
        r'(?:EDUCATION(?:AL)?\s*(?:QUALIFICATION|BACKGROUND)?|ACADEMIC\s*(?:QUALIFICATION)?)'
        r'[:\s]*\n?(.+?)(?:\nPROFESSIONAL|\nWORK|\nTECHNICAL|\nSKILLS|\nCERTIFI|\nEXPERIENCE|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if edu_match:
        edu_section = edu_match.group(1)
        lines = [l.strip() for l in edu_section.split('\n') if l.strip()]
        
        for line in lines:
            if re.match(r'^(Worked|Working|•|-|\*|PROFESSIONAL)', line, re.IGNORECASE):
                continue
            
            entry = {}
            
            # Format: "Degree | Institution | Date"
            if line.count('|') >= 2:
                parts = [p.strip() for p in line.split('|')]
                entry['degree'] = parts[0]
                entry['institution'] = parts[1]
                year_match = re.search(r'(\d{4})', parts[-1])
                entry['year'] = year_match.group(1) if year_match else None
            
            # Format: "Degree – Institution | Year-Year"
            elif '|' in line and ('–' in line or '-' in line):
                dash_match = re.match(r'^(.+?)\s*[-–]\s*(.+?)\s*\|\s*(\d{4})\s*[-–]\s*(\d{4})', line)
                if dash_match:
                    entry['degree'] = dash_match.group(1).strip()
                    entry['institution'] = dash_match.group(2).strip()
                    entry['year'] = dash_match.group(4)
            
            # Format: "Degree From Institution Year"
            elif re.search(r'\bfrom\b', line, re.IGNORECASE):
                from_match = re.match(r'^(.+?)\s+[Ff]rom\s+(.+?)\s+(\d{4})', line)
                if from_match:
                    entry['degree'] = from_match.group(1).strip()
                    entry['institution'] = from_match.group(2).strip()
                    entry['year'] = from_match.group(3)
            
            # Simple degree line
            else:
                degree_patterns = [r'\bmaster', r'\bbachelor', r'\bmba\b', r'\bmca\b', 
                                  r'\bb\.?tech\b', r'\bm\.?tech\b', r'\bb\.?e\b', r'\bm\.?e\b']
                if any(re.search(p, line.lower()) for p in degree_patterns):
                    parts = re.split(r'\s*[-–]\s*', line, maxsplit=1)
                    entry['degree'] = parts[0].strip()
                    if len(parts) > 1:
                        year_match = re.search(r'(\d{4})\s*$', parts[1])
                        if year_match:
                            entry['year'] = year_match.group(1)
                            entry['institution'] = parts[1][:year_match.start()].strip().rstrip(',|')
                        else:
                            entry['institution'] = parts[1].strip()
            
            if entry.get('degree'):
                degree_key = entry['degree'].lower()[:50]
                if degree_key not in seen_degrees:
                    seen_degrees.add(degree_key)
                    education.append({
                        'degree': entry.get('degree'),
                        'institution': entry.get('institution'),
                        'year': entry.get('year')
                    })
    
    return education


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  EXTRACTION AGENT - CERTIFICATIONS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_certifications(text: str) -> List[str]:
    """Extract certifications from resume text."""
    certifications = []
    
    cert_match = re.search(
        r'CERTIFICATIONS?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\nEDUCATION|\nSKILLS|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if cert_match:
        for line in cert_match.group(1).split('\n'):
            line = re.sub(r'^[•·\-\*]\s*', '', line.strip())
            if line and 3 < len(line) < 200:
                if re.match(r'^(PROFESSIONAL|EXPERIENCE|IMG|IBM|Cognizant)', line, re.IGNORECASE):
                    continue
                line = re.sub(r'\s*[-–]\s*In Progress\.?$', '', line, flags=re.IGNORECASE)
                certifications.append(line)
    
    return list(dict.fromkeys(certifications))[:20]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     EXTRACTION AGENT - SKILLS                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_skills(text: str) -> Dict[str, List[str]]:
    """Extract and categorize skills from resume text."""
    found_skills = {cat: [] for cat in SKILL_CATEGORIES}
    text_lower = text.lower()
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                if len(skill) <= 4 and skill.lower() not in ['go', 'r']:
                    display = skill.upper()
                elif skill in ['node.js', 'asp.net', 'vb.net', '.net']:
                    display = skill
                else:
                    display = skill.title()
                if display not in found_skills[category]:
                    found_skills[category].append(display)
    
    return {k: v for k, v in found_skills.items() if v}


def calculate_skill_experience(text: str, experiences: List[Dict]) -> Dict[str, List[Dict]]:
    """Calculate skill experience months based on job mentions."""
    skill_months: Dict[str, int] = {}
    found_skills: Dict[str, str] = {}
    text_lower = text.lower()
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                display = skill.upper() if len(skill) <= 4 and skill.lower() not in ['go', 'r'] else skill.title()
                found_skills[skill] = display
                skill_months[skill] = 0
    
    for exp in experiences:
        exp_text = ' '.join([
            str(exp.get('title', '')),
            str(exp.get('employer', '')),
            ' '.join(exp.get('responsibilities', []))
        ]).lower()
        
        for skill in found_skills:
            if re.search(rf'\b{re.escape(skill)}\b', exp_text):
                skill_months[skill] += exp.get('duration_months', 0)
    
    result = {cat: [] for cat in SKILL_CATEGORIES}
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill in found_skills:
                result[category].append({
                    'skill': found_skills[skill],
                    'experience_months': skill_months.get(skill, 0)
                })
        result[category].sort(key=lambda x: x['experience_months'], reverse=True)
    
    return {k: v for k, v in result.items() if v}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                EXTRACTION AGENT - TITLE & SUMMARY                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_title(text: str, experiences: List[Dict]) -> str:
    """Extract professional title from summary or experience."""
    summary_match = re.search(
        r'(?:PROFESSIONAL\s+)?SUMMARY[:\s]*\n(.+?)(?:\n[A-Z]{2,}|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if summary_match:
        summary = summary_match.group(1)
        
        title_match = re.match(
            r'^([\w\s/]+(?:Manager|Engineer|Developer|Analyst|Consultant|Architect|Lead))\s+with\s+\d+',
            summary.strip(), re.IGNORECASE
        )
        if title_match:
            return title_match.group(1).strip()
        
        title_match = re.search(
            r'\d+\+?\s+years?\s+(?:of\s+)?experience\s+(?:as\s+(?:a|an)\s+|in\s+)([\w\s]+?)(?:\.|,|and)',
            summary, re.IGNORECASE
        )
        if title_match:
            title = title_match.group(1).strip()
            if any(kw in title.lower() for kw in ['manager', 'engineer', 'developer', 'analyst']):
                return title
    
    if experiences and experiences[0].get('title'):
        return experiences[0]['title']
    
    return ""


def extract_summary(text: str) -> str:
    """Extract professional summary."""
    patterns = [
        r'(?:PROFESSIONAL\s+)?SUMMARY[:\s]*\n(.+?)(?:\nSKILLS|\nEXPERIENCE|\nWORK|\nTECHNICAL|\Z)',
        r'EXPERIENCE\s+SUMMARY[:\s]*\n(.+?)(?:\nTECHNICAL|\nSKILLS|\nPROFESSIONAL|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            summary = match.group(1).strip()
            summary = re.sub(r'\n+', ' ', summary)
            return summary[:2000]
    
    return ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                          VALIDATION AGENT                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def validation_agent(parsed: Dict, text: str) -> ValidationResult:
    """
    Validate parsed result quality and identify issues.
    Score: 100 (perfect) to 0 (failed)
    Triggers AI enhancement when score < 60 or critical fields missing.
    """
    result = ValidationResult()
    pr = parsed.get("parsed_resume", {})
    
    # Critical checks (20 points each)
    critical = [
        ("name", pr.get("name") and len(pr.get("name", "")) > 2),
        ("email", bool(pr.get("email") and "@" in pr.get("email", ""))),
        ("experience", len(pr.get("experience", [])) > 0),
    ]
    
    for field, passed in critical:
        if not passed:
            result.issues.append(f"missing_{field}")
            result.missing_fields.append(field)
            result.score -= 20
    
    # Important checks (10 points each)
    important = [
        ("phone", bool(pr.get("phone_number"))),
        ("title", bool(pr.get("title"))),
        ("education", len(pr.get("education", [])) > 0),
    ]
    
    for field, passed in important:
        if not passed:
            result.issues.append(f"missing_{field}")
            result.missing_fields.append(field)
            result.score -= 10
    
    # Experience quality
    experiences = pr.get("experience", [])
    if experiences:
        total_resp = sum(len(e.get("responsibilities", [])) for e in experiences)
        if total_resp < 5:
            result.issues.append("low_responsibilities")
            result.score -= 15
        
        missing_titles = sum(1 for e in experiences if not e.get("title"))
        missing_employers = sum(1 for e in experiences if not e.get("Employer"))
        
        result.score -= missing_titles * 5
        result.score -= missing_employers * 5
    
    # Name sanity check
    name = (pr.get("name") or "").lower()
    invalid = ["sql server", "data engineer", "project manager", "key expertise", "professional"]
    if any(inv in name for inv in invalid):
        result.issues.append("invalid_name")
        result.missing_fields.append("name")
        result.score -= 25
    
    result.needs_ai_enhancement = result.score < 60 or "missing_name" in result.issues or "invalid_name" in result.issues
    result.score = max(0, result.score)
    
    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                        AI ENHANCEMENT AGENT                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def ai_enhancement_agent(text: str, parsed: Dict, validation: ValidationResult) -> Dict:
    """
    Use Claude API to extract/fix missing fields.
    Only called when validation score is low or critical fields missing.
    """
    if not ANTHROPIC_API_KEY:
        parsed["ai_skipped"] = "No API key configured"
        return parsed
    
    try:
        import httpx
        
        missing = list(set(validation.missing_fields))
        
        prompt = f"""You are a resume parsing expert. Extract these MISSING fields from this resume.
Return ONLY valid JSON with the requested fields, no markdown or explanation.

MISSING FIELDS: {', '.join(missing)}

RESUME TEXT:
{text[:15000]}

Return JSON format:
{{
  "firstname": "First name",
  "lastname": "Last name", 
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1234567890",
  "title": "Professional Title",
  "education": [{{"degree": "...", "institution": "...", "year": "YYYY"}}],
  "experience": [{{"Employer": "Company", "title": "Job Title", "start_date": "YYYY-MM", "end_date": "YYYY-MM", "duration_months": N, "responsibilities": ["..."]}}]
}}"""

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 8000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                ai_text = response.json().get("content", [{}])[0].get("text", "")
                json_match = re.search(r'\{[\s\S]*\}', ai_text)
                
                if json_match:
                    ai_data = json.loads(json_match.group())
                    pr = parsed.get("parsed_resume", {})
                    
                    for key in ['firstname', 'lastname', 'name', 'email', 'title']:
                        if key in missing and ai_data.get(key):
                            pr[key] = ai_data[key]
                    
                    if 'phone' in missing and ai_data.get('phone'):
                        pr['phone_number'] = ai_data['phone']
                    
                    if 'education' in missing and ai_data.get('education'):
                        pr['education'] = ai_data['education']
                    
                    if 'experience' in missing and ai_data.get('experience'):
                        if not pr.get('experience'):
                            pr['experience'] = ai_data['experience']
                    
                    parsed['ai_enhanced'] = True
                    parsed['ai_fields_fixed'] = missing
            else:
                parsed['ai_error'] = f"API returned {response.status_code}"
                    
    except Exception as e:
        parsed['ai_error'] = str(e)
    
    return parsed


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           OUTPUT AGENT                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

async def parse_resume(text: str, filename: str = None, use_ai: bool = True) -> Dict:
    """
    Orchestrate all agents and produce final standardized output.
    Flow: Extraction → Validation → AI Enhancement → JSON Output
    """
    text = normalize_text(text)
    
    # ===== EXTRACTION AGENT =====
    contact = extract_contact(text)
    firstname, middle, lastname = extract_name(text)
    experiences = extract_experiences(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    skills = extract_skills(text)
    skill_experience = calculate_skill_experience(text, experiences)
    summary = extract_summary(text)
    
    # Build full name
    name_parts = [firstname]
    if middle:
        name_parts.append(middle)
    name_parts.append(lastname)
    name = ' '.join(filter(None, name_parts))
    
    # Extract title
    title = extract_title(text, experiences)
    
    # Calculate totals
    total_months = sum(e.get('duration_months', 0) for e in experiences)
    
    # Build result
    result = {
        "parsed_resume": {
            "firstname": firstname or None,
            "lastname": lastname or None,
            "name": name or None,
            "title": title or None,
            "location": contact.get('location') or None,
            "phone_number": contact.get('phone') or None,
            "email": contact.get('email') or None,
            "linkedin": contact.get('linkedin') or None,
            "summary": summary or None,
            "total_experience_months": total_months,
            "total_experience_years": round(total_months / 12, 1) if total_months else 0,
            "technical_skills": [s for cat in skills.values() for s in cat],
            "key_skills": skill_experience,
            "education": education,
            "certifications": certifications,
            "experience": [
                {
                    "Employer": e.get('employer'),
                    "title": e.get('title'),
                    "location": e.get('location'),
                    "start_date": e.get('start_date'),
                    "end_date": e.get('end_date'),
                    "duration_months": e.get('duration_months'),
                    "responsibilities": e.get('responsibilities', [])
                }
                for e in experiences
            ],
            "filename": filename or ""
        },
        "parser_version": VERSION
    }
    
    # ===== VALIDATION AGENT =====
    validation = validation_agent(result, text)
    result["validation_score"] = validation.score
    result["validation_issues"] = validation.issues
    
    # ===== AI ENHANCEMENT AGENT =====
    if use_ai and validation.needs_ai_enhancement and ANTHROPIC_API_KEY:
        result = await ai_enhancement_agent(text, result, validation)
    
    return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           API ENDPOINTS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@app.get("/", tags=["Info"])
async def root():
    """API information and capabilities."""
    return {
        "name": "Resume Parser API",
        "version": VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "ai_validation": "enabled" if ANTHROPIC_API_KEY else "disabled",
        "supported_formats": ["pdf", "docx", "txt", "zip"],
        "architecture": "Agentic Framework",
        "agents": [
            "Extraction Agent (7 patterns)",
            "Validation Agent (quality scoring)",
            "AI Enhancement Agent (Claude API)",
            "Output Agent (JSON generation)"
        ],
        "patterns": [
            "1. Standard 'Company – Title Date'",
            "2. 'Worked as X in Y from A to B'",
            "3. Table 'Client: X | Duration: Y'",
            "4. 'Title (Date) – Client – Employer'",
            "5. 'Company Date' then 'Title' next line",
            "6. Pipe 'Title | Date' with company above",
            "7. 'ROLE:' / 'DESIGNATION:' keyword"
        ]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ai_available": bool(ANTHROPIC_API_KEY),
        "version": VERSION
    }


@app.post("/parse/file", tags=["Parsing"])
async def parse_file(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, TXT)"),
    use_ai_validation: bool = Form(default=True, description="Enable AI enhancement")
):
    """
    Parse resume from uploaded file.
    
    Supports:
    - PDF files (including multi-column layouts)
    - DOCX files (with tables and text boxes)
    - TXT files
    - ZIP archives containing text files
    """
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(400, "Empty file")
        
        text = extract_text_intelligent(content, file.filename)
        
        if len(text.strip()) < 50:
            file_type = detect_file_type(content, file.filename)
            raise HTTPException(
                400, 
                f"Insufficient text extracted ({len(text)} chars). "
                f"Detected file type: {file_type}"
            )
        
        result = await parse_resume(text, file.filename, use_ai_validation)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")


@app.post("/parse", tags=["Parsing"])
async def parse_text(request: ParseTextRequest):
    """Parse resume from text input."""
    try:
        result = await parse_resume(
            request.text,
            request.filename,
            request.use_ai_validation
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.post("/extract/skills", tags=["Utilities"])
async def extract_skills_endpoint(request: ExtractSkillsRequest):
    """Extract and categorize technical skills from text."""
    skills = extract_skills(request.text)
    flat_skills = [s for cat in skills.values() for s in cat]
    return {
        "skills": flat_skills, 
        "categorized": skills, 
        "count": len(flat_skills),
        "categories": list(skills.keys())
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MAIN ENTRY POINT                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8080))
    
    print("╔" + "═" * 70 + "╗")
    print(f"║{'RESUME PARSER API v' + VERSION + ' - Enterprise Agentic Framework':^70}║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  Port: {port:<62}║")
    print(f"║  AI Enhancement: {'Enabled' if ANTHROPIC_API_KEY else 'Disabled':<52}║")
    print(f"║  Docs: http://localhost:{port}/docs{' ' * 37}║")
    print("╠" + "═" * 70 + "╣")
    print("║  Architecture:                                                        ║")
    print("║    ┌─────────────────┐                                                ║")
    print("║    │ Extraction Agent│ ──▶ 7 Pattern-based extraction                 ║")
    print("║    └────────┬────────┘                                                ║")
    print("║             ▼                                                         ║")
    print("║    ┌─────────────────┐                                                ║")
    print("║    │ Validation Agent│ ──▶ Quality scoring (0-100)                    ║")
    print("║    └────────┬────────┘                                                ║")
    print("║             ▼                                                         ║")
    print("║    ┌─────────────────┐                                                ║")
    print("║    │ AI Enhancement  │ ──▶ Claude API (if needed)                     ║")
    print("║    └────────┬────────┘                                                ║")
    print("║             ▼                                                         ║")
    print("║    ┌─────────────────┐                                                ║")
    print("║    │ Output Agent    │ ──▶ Standardized JSON                          ║")
    print("║    └─────────────────┘                                                ║")
    print("╚" + "═" * 70 + "╝")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
