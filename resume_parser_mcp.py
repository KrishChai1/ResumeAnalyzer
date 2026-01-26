"""
Resume Parser MCP Server - Enterprise Grade v2.1
=================================================
Google/LinkedIn-level resume parsing with comprehensive extraction.

Features:
- Multi-format support (PDF, DOCX, TXT)
- Intelligent section detection
- Comprehensive experience extraction with responsibilities
- Skill-to-experience mapping with duration calculation
- Education extraction from multiple formats
- Claude AI validation for complex cases

Configuration:
- Set ANTHROPIC_API_KEY environment variable for AI enhancement
"""

import json
import re
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
import unicodedata

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("resume_parser_mcp")
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp = None


# ============================================================================
# CONFIGURATION
# ============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ============================================================================
# CONSTANTS
# ============================================================================

MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

SKILL_CATEGORIES = {
    "Data Engineering": [
        "etl", "data migration", "data pipelines", "informatica", "ssis", "talend",
        "apache airflow", "airflow", "apache spark", "spark", "pyspark",
        "apache kafka", "kafka", "snowflake", "databricks", "dbt", "glue", "redshift",
        "data warehousing", "dwh", "olap", "oltp", "star schema"
    ],
    "Programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "cobol", "bash", "shell",
        "powershell", "sql", "plsql", "html", "css", "node.js", "nodejs", "react",
        "angular", "vue", "django", "flask", ".net", "asp.net", "vb.net", "gherkin",
        "scripting", "automation scripting"
    ],
    "Databases": [
        "mongodb", "cassandra", "redis", "postgresql", "mysql", "oracle", "db2",
        "sql server", "nosql", "dynamodb", "elasticsearch", "neo4j", "sql developer",
        "database", "database management"
    ],
    "Cloud": [
        "aws", "azure", "gcp", "google cloud", "ibm cloud", "openstack",
        "ec2", "s3", "lambda", "ecs", "ecr", "eks", "rds", "vpc", "route 53",
        "multi-cloud", "hybrid cloud", "cloud computing", "cloud security",
        "vmware", "vmware vsphere", "vcloud", "virtualization", "hyper-v"
    ],
    "DevOps": [
        "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
        "jenkins", "gitlab", "github actions", "ci/cd", "cicd", "concourse",
        "helm", "gitops", "devops", "devsecops", "infrastructure as code",
        "azure devops", "agile", "waterfall", "scrum", "kanban"
    ],
    "Testing & QA": [
        "selenium", "pytest", "unittest", "testng", "junit", "cucumber", "behave",
        "bdd", "tdd", "manual testing", "automation testing", "regression testing",
        "api testing", "postman", "functional testing", "uat", "quality assurance"
    ],
    "Security": [
        "sonarqube", "qualys", "crowdstrike", "mend", "uptycs", "snyk",
        "siem", "splunk", "palo alto", "firewall", "iam", "pam", "mfa",
        "cisco security", "checkpoint", "fortinet", "zscaler", "risk & compliance"
    ],
    "Data Science & Visualization": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "pandas", "numpy", "tableau", "power bi", "grafana", "prometheus",
        "elk stack", "kibana", "data visualization", "bi reporting",
        "executive dashboards", "sla reporting"
    ],
    "Other Tools": [
        "jira", "confluence", "servicenow", "git", "github", "linux",
        "unix", "rhel", "ubuntu", "centos", "windows server",
        "putty", "pycharm", "hp alm", "zephyr", "ms project", "clarity-ppm",
        "power automate", "itil", "pmo governance", "risk management",
        "project management", "program management", "portfolio management",
        "business continuity", "disaster recovery"
    ],
    "Business Domains": [
        "banking", "finance", "financial services", "telecom", "telecommunications",
        "healthcare", "pharma", "pharmaceuticals", "insurance", "retail", "e-commerce",
        "crm", "bss", "oss", "billing", "payment", "aviation", "education", "publishing"
    ]
}

ALL_SKILLS: Set[str] = set()
for skills in SKILL_CATEGORIES.values():
    ALL_SKILLS.update(skills)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ExperienceEntry:
    employer: str
    title: str
    location: str
    start_date: str
    end_date: str
    duration_months: int
    responsibilities: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    project: str = ""
    client: str = ""


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class ParseResumeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    resume_text: str = Field(..., min_length=50, max_length=100000)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON)
    filename: Optional[str] = Field(default=None)
    use_ai_validation: bool = Field(default=True)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:
    """Clean and normalize resume text - enterprise grade."""
    if not text:
        return ""
    
    # Fix encoding issues (â€" is a common encoding problem for em-dash)
    text = text.replace('â€"', '–').replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    text = text.replace('Ã©', 'é').replace('Ã¨', 'è').replace('Ã ', 'à')
    
    # Normalize unicode
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('–', '-')
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '•').replace('\u00a0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
    
    # Fix common PDF issues
    text = re.sub(r'Pres\s*ent', 'Present', text, flags=re.IGNORECASE)
    text = re.sub(r'US\s*A', 'USA', text)
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_text_from_docx_with_tables(file_path: str) -> str:
    """Extract text from DOCX including tables - handles all formats."""
    from docx import Document
    doc = Document(file_path)
    
    all_text = []
    
    # Extract paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            all_text.append(text)
    
    # Extract tables (important for resumes with tabular format)
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                # Join cells with | separator
                all_text.append(' | '.join(row_text))
    
    return '\n'.join(all_text)


def clean_output_text(text: str) -> str:
    """Clean text for final output - remove artifacts."""
    if not text:
        return ""
    # Remove literal \n
    text = text.replace('\\n', ' ')
    # Fix encoding artifacts
    text = text.replace('â€"', '-').replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
    # Remove location patterns embedded in titles
    text = re.sub(r'\s+(Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]*(PA|TX|IL|NY|CA|USA|India|Karnataka)?\s*$', '', text, flags=re.IGNORECASE)
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ============================================================================
# CONTACT EXTRACTION
# ============================================================================

def extract_contact(text: str) -> Dict[str, str]:
    """Extract contact information."""
    contact = {'email': '', 'phone': '', 'linkedin': '', 'location': ''}
    
    # Email
    match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text)
    if match:
        contact['email'] = match.group(1).lower()
    
    # Phone
    phone_patterns = [
        r'(?:Mob|Phone|Tel|Mobile)[:\s]*(\+?[\d\s\-().]{10,})',
        r'(\+1\s*\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
        r'(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
        r'(\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
        r'(\d{10})',
        r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phone = re.sub(r'[^\d+\-() ]', '', match.group(1)).strip()
            if len(re.sub(r'\D', '', phone)) >= 10:
                contact['phone'] = phone
                break
    
    # LinkedIn - multiple patterns
    linkedin_patterns = [
        r'linkedin\.com/in/([\w-]+)',
        r'linkedin[:\s]+(?:www\.)?linkedin\.com/in/([\w-]+)',
        r'LinkedIn[:\s]+([\w]+)(?:\s|$)',
    ]
    for pattern in linkedin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            username = match.group(1)
            # Skip if it's a generic word
            if username.lower() not in ['summary', 'profile', 'in', 'phone', 'email']:
                contact['linkedin'] = f"www.linkedin.com/in/{username}"
                break
    
    # Location
    loc_patterns = [
        r'(Philadelphia|Chicago|Dallas|Plano|New York|Bangalore|Bengaluru|Hyderabad|Pune|Mumbai)[,\s]+(PA|TX|IL|NY|CA|India|USA|Karnataka|Maharashtra)',
        r'\b([A-Z][a-z]+),\s*([A-Z]{2})\b',
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, text)
        if match and not contact['location']:
            contact['location'] = f"{match.group(1)}, {match.group(2)}"
            break
    
    return contact


# ============================================================================
# NAME & TITLE EXTRACTION
# ============================================================================

def extract_name(text: str) -> Tuple[str, str, str]:
    """Extract first, middle, last name."""
    lines = [l.strip() for l in text.split('\n') if l.strip()][:25]
    
    # Headers and keywords to skip (only if they appear at the START of line)
    skip_start_patterns = [
        'resume', 'cv', 'curriculum vitae', 'key expertise', 'additional details',
        'professional experience', 'education', 'skills', 'summary', 'objective',
        'technical', 'core competencies', 'profile', 'experience summary',
        'achievements', 'certifications', 'professional summary', 'experience -',
        'data warehouse', 'sql server', 'reporting tools', 'results-driven',
        'experienced', 'skilled', 'dedicated', 'professional with', 'having'
    ]
    
    for line in lines:
        # Clean the line first (remove excessive whitespace)
        clean_line = ' '.join(line.split())
        
        # Remove contact info suffixes BEFORE checking skip patterns
        # "NAVEEN REDDY YADLA Contact: 470-505-9469" -> "NAVEEN REDDY YADLA"
        name_part = re.sub(r'\s*(Contact|Phone|Email|Tel|Cell|Mobile)[:\s].*$', '', clean_line, flags=re.IGNORECASE).strip()
        
        # Skip lines that START with skip patterns
        if any(name_part.lower().startswith(skip) for skip in skip_start_patterns):
            continue
        # Skip lines ending with colon (likely headers)
        if name_part.endswith(':'):
            continue
        # Skip if entire line is an email or phone
        if re.match(r'^[\w.+-]+@[\w.-]+\.\w+$', name_part) or re.match(r'^[\d\s\-+()]+$', name_part):
            continue
        
        # Clean name further - remove email/phone if inline
        name = name_part
        name = re.sub(r'[\|].*$', '', name).strip()
        name = re.sub(r'\s*Email:.*$', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'\s*Cell:.*$', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'\s*Phone:.*$', '', name, flags=re.IGNORECASE).strip()
        name = re.sub(r'\s*,.*$', '', name).strip()  # Remove anything after comma
        
        parts = name.split()
        
        # Check if it looks like a name (2-4 parts, all start with uppercase)
        if 2 <= len(parts) <= 4:
            if all(p[0].isupper() for p in parts if p):
                # Additional check: not tech terms
                tech_terms = ['SQL', 'ETL', 'GCP', 'AWS', 'API', 'XML', 'JSON', 'HTML', 'CSS', 'SSIS', 'SSAS']
                if not any(p in tech_terms for p in parts):
                    if len(parts) == 2:
                        return parts[0], "", parts[1]
                    elif len(parts) == 3:
                        return parts[0], parts[1], parts[2]
                    else:
                        return parts[0], ' '.join(parts[1:-1]), parts[-1]
    
    return "", "", ""


def extract_title(text: str, experiences: List[ExperienceEntry]) -> str:
    """Extract professional title from summary (preferred) or experience."""
    
    # Try from PROFESSIONAL SUMMARY first (most accurate)
    summary_match = re.search(
        r'(?:PROFESSIONAL\s+)?SUMMARY[:\s]*\n(.+?)(?:\n[A-Z]{2,}|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if summary_match:
        summary = summary_match.group(1)
        
        # Pattern 1: "Title with X+ years" at the start
        # Example: "Project Manager with 18+ years of experience"
        title_match = re.match(
            r'^([\w\s/]+(?:Manager|Engineer|Developer|Analyst|Consultant|Architect|Lead|Specialist|Director|Administrator))\s+with\s+\d+',
            summary.strip(), re.IGNORECASE
        )
        if title_match:
            return clean_output_text(title_match.group(1))
        
        # Pattern 2: "X years of experience as/in Title"
        title_match = re.search(
            r'\d+\+?\s+years?\s+(?:of\s+)?experience\s+(?:as\s+(?:a|an)\s+|in\s+)([\w\s]+?)(?:\.|,|and)',
            summary, re.IGNORECASE
        )
        if title_match:
            title = title_match.group(1).strip()
            if any(kw in title.lower() for kw in ['manager', 'engineer', 'developer', 'analyst', 'consultant', 'lead', 'tester']):
                return clean_output_text(title)
    
    # Fallback to most recent job title (cleaned)
    if experiences:
        title = experiences[0].title
        if title:
            # Remove suffix like "– Cognizant Infra Services"
            title = re.sub(r'\s*[-–]\s*[A-Z][\w\s]+(?:Services|Solutions|Group|Team)?\s*$', '', title)
            # Remove location
            title = re.sub(r'\s+(Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago|India|USA|Karnataka)[,\s]*(?:PA|TX|IL|NY|CA|USA|India)?$', '', title, flags=re.IGNORECASE)
            return clean_output_text(title)
    
    return ""


# ============================================================================
# PATTERN DOCUMENTATION - SUPPORTED RESUME FORMATS
# ============================================================================
"""
EXPERIENCE PATTERNS SUPPORTED:
==============================

Pattern 1: PIPE FORMAT (Jimmy style)
    Company – Location
    Title | Date Range
    • Responsibilities...
    
Pattern 2: INLINE FORMAT (Sudheer style)
    Company – Title Location Date Range
    • Responsibilities...
    
Pattern 3: WORKED AS FORMAT (Madhuri style)
    "Worked as Title in Company from Month Year to Month Year"
    (Dates in summary, responsibilities in separate WORK EXPERIENCE section)
    
Pattern 4: COMPANY-DATE LINE (Khaliq style)
    Company Date Range
    Title Location
    • Responsibilities...

Pattern 5: CLIENT FORMAT (Sudheer TCS style)
    Company – Title, Location Client – ClientName – NA Date Range
    • Responsibilities...

EDUCATION PATTERNS SUPPORTED:
=============================

Pattern 1: PIPE FORMAT (Sudheer style)
    Degree | Institution | Month Year

Pattern 2: DASH-PIPE FORMAT (Jimmy style)
    Degree – Institution | Year-Year

Pattern 3: INSTITUTION-THEN-DEGREE (Khaliq style)
    Institution Date Range
    Degree

Pattern 4: FROM FORMAT (Madhuri style)
    Degree From Institution Year

CONTACT PATTERNS SUPPORTED:
===========================
- Email: standard@domain.com
- Phone: +1(469)781-4257, (469) 629-9205, +1 469 867 5274, 7038034869
- LinkedIn: linkedin.com/in/username, LinkedIn: username
- Location: City, State/Country patterns
"""

# ============================================================================
# MULTI-STRATEGY EXTRACTION
# ============================================================================

def extract_experiences_multi_strategy(text: str) -> List[ExperienceEntry]:
    """
    Try multiple extraction strategies and return the best result.
    This ensures we handle any resume format without breaking.
    """
    strategies = [
        ("worked_as", extract_experiences_worked_as),
        ("date_line", extract_experiences_date_line),
    ]
    
    all_results = []
    
    for strategy_name, strategy_func in strategies:
        try:
            results = strategy_func(text)
            if results:
                # Score based on completeness
                score = sum(
                    (1 if e.employer else 0) +
                    (1 if e.title else 0) +
                    (2 if e.responsibilities else 0) +
                    (1 if e.duration_months > 0 else 0)
                    for e in results
                )
                all_results.append((strategy_name, results, score))
        except Exception:
            continue
    
    if not all_results:
        return []
    
    # Return the strategy with highest score
    all_results.sort(key=lambda x: x[2], reverse=True)
    return all_results[0][1]


def extract_experiences_worked_as(text: str) -> List[ExperienceEntry]:
    """Strategy 1: Extract from 'Worked as X in Y from A to B' format."""
    experiences = []
    
    worked_pattern = r'[Ww]ork(?:ed|ing)\s+(?:as\s+)?(?:a\s+)?(.+?)\s+in\s+(.+?)\s+from\s+(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|Present|Current)'
    
    for match in re.finditer(worked_pattern, text, re.IGNORECASE):
        title = match.group(1).strip()
        employer = match.group(2).strip()
        start_year, start_month, _ = parse_date(match.group(3))
        end_year, end_month, is_present = parse_date(match.group(4))
        
        if start_year and end_year:
            duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
            start_date = f"{start_year}-{(start_month or 1):02d}"
            end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
            
            experiences.append(ExperienceEntry(
                employer=employer,
                title=title,
                location="",
                start_date=start_date,
                end_date=end_date,
                duration_months=duration
            ))
    
    return experiences


def extract_experiences_date_line(text: str) -> List[ExperienceEntry]:
    """Strategy 2: Extract from lines containing date ranges."""
    experiences = []
    
    date_range_pattern = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current)'
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        date_match = re.search(date_range_pattern, line, re.IGNORECASE)
        
        if date_match:
            start_str, end_str = date_match.group(1), date_match.group(2)
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                employer, title, location, client = "", "", "", ""
                
                # Try to parse the line before the date
                header = line[:date_match.start()].strip()
                
                # Different parsing strategies based on line format
                employer, title, location, client = parse_experience_header(header, lines, i)
                
                # Get responsibilities
                responsibilities = extract_responsibilities(lines, i + 1)
                
                # Calculate duration
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                # Extract tools from responsibilities
                tools = extract_tools_from_text(' '.join(responsibilities))
                
                if employer or title:
                    # Check for duplicates
                    is_dup = any(
                        e.employer == employer and e.start_date == start_date
                        for e in experiences
                    )
                    if not is_dup:
                        experiences.append(ExperienceEntry(
                            employer=employer,
                            title=title.strip(),
                            location=location,
                            start_date=start_date,
                            end_date=end_date,
                            duration_months=duration,
                            responsibilities=responsibilities[:12],
                            tools=tools,
                            client=client
                        ))
        i += 1
    
    return experiences


def parse_experience_header(header: str, lines: List[str], current_idx: int) -> Tuple[str, str, str, str]:
    """
    Parse experience header using multiple pattern strategies.
    Returns: (employer, title, location, client)
    """
    employer, title, location, client = "", "", "", ""
    line = lines[current_idx].strip()
    
    # Check if this is a pipe format line
    if '|' in line:
        # Pattern: "Title | Date" with company on previous line
        pipe_idx = line.rfind('|')
        title_part = line[:pipe_idx].strip()
        
        # Check for client pattern
        client_match = re.search(r'Client[:\s]+([^|]+)', title_part, re.IGNORECASE)
        if client_match:
            client = client_match.group(1).strip()
        
        # Full title (including suffix like "– Cognizant Infra Services")
        title = title_part.split('|')[0].strip()
        
        # Previous line is company
        if current_idx > 0:
            prev_line = lines[current_idx - 1].strip()
            loc_match = re.search(r'[-–]\s*(.+)$', prev_line)
            if loc_match:
                employer = prev_line[:prev_line.find('-')].strip()
                location = loc_match.group(1).strip()
            else:
                employer = prev_line
    else:
        # Non-pipe format
        
        # Pattern 1: "Company – Title Location"
        comp_match = re.match(
            r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+?)\s+((?:Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago|India)[,\s]*(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)\s*$',
            header
        )
        if comp_match:
            employer = comp_match.group(1).strip()
            title = comp_match.group(2).strip()
            location = comp_match.group(3).strip()
        else:
            # Pattern 2: "Company – Title, Location Client – ClientName"
            dash_match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+)$', header)
            if dash_match:
                employer = dash_match.group(1).strip()
                title_part = dash_match.group(2).strip()
                
                # Check for client pattern
                client_match = re.search(r'\s*Client\s*[-–]\s*(.+?)(?:\s*[-–]\s*(?:NA|N/A))?\s*$', title_part, re.IGNORECASE)
                if client_match:
                    title = title_part[:client_match.start()].strip().rstrip(',')
                    client = client_match.group(1).strip()
                else:
                    title = title_part
                
                # Extract location if embedded
                loc_match = re.search(r',?\s*(India|USA|Philadelphia|Chicago|Bangalore)\s*$', title, re.IGNORECASE)
                if loc_match:
                    location = loc_match.group(1).strip()
                    title = title[:loc_match.start()].strip().rstrip(',')
            else:
                # Pattern 3: Company only, title on next line
                employer = header.strip()
                if current_idx + 1 < len(lines):
                    date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}'
                    next_line = lines[current_idx + 1].strip()
                    if not re.search(date_pattern, next_line) and not next_line.startswith(('•', '-', '*')):
                        loc_match = re.search(
                            r'((?:Philadelphia|Chicago|Bangalore|Hyderabad)[,\s]*(?:PA|TX|India|Karnataka)?)\s*$',
                            next_line, re.IGNORECASE
                        )
                        if loc_match:
                            title = next_line[:loc_match.start()].strip()
                            location = loc_match.group(1).strip()
                        else:
                            title = next_line
        
        # Final cleanup: if employer still contains title
        if not title and '–' in employer:
            parts = employer.split('–', 1)
            employer = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else ""
    
    return employer, title, location, client


def extract_responsibilities(lines: List[str], start_idx: int) -> List[str]:
    """Extract responsibilities starting from given index."""
    responsibilities = []
    date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\s*[-–]'
    
    j = start_idx
    while j < len(lines) and j < start_idx + 30:
        resp_line = lines[j].strip()
        
        # Stop conditions
        if re.search(date_pattern, resp_line, re.IGNORECASE):
            break
        if re.match(r'^(EDUCATION|TECHNICAL|SKILLS|CERTIFICATIONS|PERSONAL|KEY\s+ARCH|PROFESSIONAL\s+EXPERIENCE)', resp_line, re.IGNORECASE):
            break
        if re.match(r'^[A-Z][A-Za-z\s&]+(?:Ltd|Inc|Corp|Technologies|Solutions)?\s*[-–]\s*[A-Z]', resp_line):
            break
        
        # Extract bullet points
        if resp_line.startswith(('•', '-', '*', '–')) or re.match(r'^\d+\.', resp_line):
            resp_text = re.sub(r'^[•\-\*–\d.]\s*', '', resp_line)
            if len(resp_text) > 20:
                responsibilities.append(clean_output_text(resp_text))
        
        j += 1
    
    return responsibilities

def parse_date(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse date string to (year, month, is_present)."""
    if not text:
        return None, None, False
    
    text = text.strip().lower()
    
    if any(p in text for p in ['present', 'current', 'now', 'ongoing', 'till date']):
        now = datetime.now()
        return now.year, now.month, True
    
    # Month Year (full or abbreviated)
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*[,.]?\s*(\d{4})', text)
    if match:
        month = MONTH_MAP.get(match.group(1)[:3])
        year = int(match.group(2))
        return year, month, False
    
    # Compact format: Jan2021, Jul24
    match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2,4})', text, re.IGNORECASE)
    if match:
        month = MONTH_MAP.get(match.group(1).lower())
        year_str = match.group(2)
        if len(year_str) == 2:
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        else:
            year = int(year_str)
        return year, month, False
    
    # Just year
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if match:
        return int(match.group(1)), None, False
    
    return None, None, False


def parse_short_date(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse short date format like Jul24, Jun21, Present, P."""
    if not text:
        return None, None, False
    
    text = text.strip().lower()
    
    # Present or P
    if text in ['present', 'p', 'current', 'now']:
        now = datetime.now()
        return now.year, now.month, True
    
    # Short format: Jul24, Jun21
    match = re.match(r'(\w{3})(\d{2})$', text, re.IGNORECASE)
    if match:
        month_str = match.group(1).lower()
        year_short = int(match.group(2))
        month = MONTH_MAP.get(month_str, 1)
        year = 2000 + year_short if year_short < 50 else 1900 + year_short
        return year, month, False
    
    # Fallback to regular parse
    return parse_date(text)


def calculate_duration(start_year: int, start_month: int, end_year: int, end_month: int) -> int:
    """Calculate duration in months (inclusive of both start and end month)."""
    start_dt = datetime(start_year, start_month or 1, 1)
    end_dt = datetime(end_year, end_month or 12, 1)
    delta = relativedelta(end_dt, start_dt)
    # Add 1 to include both start and end months
    return max(1, delta.years * 12 + delta.months + 1)


# ============================================================================
# EXPERIENCE EXTRACTION - ENTERPRISE GRADE
# ============================================================================

def extract_experiences(text: str) -> List[ExperienceEntry]:
    """Extract work experiences - handles multiple formats."""
    experiences = []
    
    # Strategy 1: "Worked as X in Y from A to B" format
    worked_pattern = r'[Ww]ork(?:ed|ing)\s+(?:as\s+)?(?:a\s+)?(.+?)\s+in\s+(.+?)\s+from\s+(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|Present|Current)'
    for match in re.finditer(worked_pattern, text, re.IGNORECASE):
        title = match.group(1).strip()
        employer = match.group(2).strip()
        start_year, start_month, _ = parse_date(match.group(3))
        end_year, end_month, is_present = parse_date(match.group(4))
        
        if start_year and end_year:
            duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
            start_date = f"{start_year}-{(start_month or 1):02d}"
            end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
            
            experiences.append(ExperienceEntry(
                employer=employer,
                title=title,
                location="",
                start_date=start_date,
                end_date=end_date,
                duration_months=duration
            ))
    
    # Strategy 2: "Title (DateRange) – Client – Employer" format (Nageswara style)
    # Example: "GCP Data Engineer (Jan2021 - Present) – Renault-Nissan – Atos Global IT Solutions"
    title_date_pattern = r'^[\s]*([\w\s]+?)\s*\((\w{3,9}\d{4})\s*[-–]\s*(\w{3,9}\d{4}|Present|Current)\)\s*[-–]\s*(.+?)[-–]\s*(.+)$'
    for line in text.split('\n'):
        match = re.match(title_date_pattern, line.strip(), re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            start_str = match.group(2)
            end_str = match.group(3)
            client = match.group(4).strip()
            employer = match.group(5).strip()
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                is_dup = any(e.title == title and e.start_date == start_date for e in experiences)
                if not is_dup:
                    experiences.append(ExperienceEntry(
                        employer=employer,
                        title=title,
                        location="",
                        start_date=start_date,
                        end_date=end_date,
                        duration_months=duration,
                        client=client
                    ))
    
    # Strategy 3: "Client: Company – Location Date" then "Title" (Naveen style)
    # Example: "Client: Ascent Global Logistics – Atlanta, GA.    Jul24 to Present"
    client_pattern = r'^Client:\s*(.+?)[-–]\s*([A-Za-z\s,]+\.?)\s+(\w{3}\d{2})\s+to\s+(\w{3}\d{2}|Present|P)$'
    lines = text.split('\n')
    for i, line in enumerate(lines):
        match = re.match(client_pattern, line.strip(), re.IGNORECASE)
        if match:
            employer = match.group(1).strip()
            location = match.group(2).strip().rstrip('.')
            start_str = match.group(3)
            end_str = match.group(4)
            
            # Convert short date format (Jul24 -> July 2024)
            start_year, start_month, _ = parse_short_date(start_str)
            end_year, end_month, is_present = parse_short_date(end_str)
            
            if start_year and end_year:
                # Next line should be title
                title = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(('Client:', 'Environment:', 'Description:', 'Responsibilities:')):
                        title = next_line
                
                # Get responsibilities
                responsibilities = []
                for j in range(i + 2, min(i + 30, len(lines))):
                    resp_line = lines[j].strip()
                    if resp_line.startswith('Client:') or resp_line.startswith('Environment:'):
                        break
                    if resp_line.startswith(('•', '-', '*')) or (resp_line and not resp_line.startswith(('Description:', 'Responsibilities:'))):
                        resp = re.sub(r'^[•\-\*]\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(clean_output_text(resp))
                
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                is_dup = any(e.employer == employer and e.start_date == start_date for e in experiences)
                if not is_dup:
                    experiences.append(ExperienceEntry(
                        employer=employer,
                        title=title,
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        duration_months=duration,
                        responsibilities=responsibilities[:12],
                        tools=extract_tools_from_text(' '.join(responsibilities))
                    ))
    
    # Strategy 4: Table-based "Client: X | Duration: Date" (Ramaswamy style)
    table_client_pattern = r'Client:\s*([^\|]+)'
    table_duration_pattern = r'Duration:\s*(\w{3,9})[-–](\d{4})\s+to\s+(\w{3,9}|\w+\s+\w+)[-–]?(\d{4})?'
    
    for line in text.split('\n'):
        client_match = re.search(table_client_pattern, line, re.IGNORECASE)
        duration_match = re.search(table_duration_pattern, line, re.IGNORECASE)
        
        if client_match and duration_match:
            employer = client_match.group(1).strip()
            start_month_str = duration_match.group(1)
            start_year = int(duration_match.group(2))
            end_str = duration_match.group(3)
            end_year_str = duration_match.group(4)
            
            start_month = MONTH_MAP.get(start_month_str[:3].lower(), 1)
            
            if 'till' in end_str.lower() or 'present' in end_str.lower() or 'date' in end_str.lower():
                end_year = datetime.now().year
                end_month = datetime.now().month
                is_present = True
            else:
                end_month = MONTH_MAP.get(end_str[:3].lower(), 12)
                end_year = int(end_year_str) if end_year_str else start_year
                is_present = False
            
            duration = calculate_duration(start_year, start_month, end_year, end_month)
            start_date = f"{start_year}-{start_month:02d}"
            end_date = f"{end_year}-{end_month:02d}"
            
            is_dup = any(e.employer == employer and e.start_date == start_date for e in experiences)
            if not is_dup:
                experiences.append(ExperienceEntry(
                    employer=employer,
                    title="",  # Will be filled by responsibilities or AI
                    location="",
                    start_date=start_date,
                    end_date=end_date,
                    duration_months=duration
                ))
    
    # Strategy 5: Standard date range detection
    date_range_pattern = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current)'
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        date_match = re.search(date_range_pattern, line, re.IGNORECASE)
        
        if date_match:
            start_str, end_str = date_match.group(1), date_match.group(2)
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                employer, title, location, client = "", "", "", ""
                
                # Check for pipe format: "Title | Date"
                if '|' in line:
                    pipe_idx = line.rfind('|')
                    title_part = line[:pipe_idx].strip()
                    
                    # Extract client if present: "Title | Client: X | Date"
                    client_match = re.search(r'Client[:\s]+([^|]+)', title_part, re.IGNORECASE)
                    if client_match:
                        client = client_match.group(1).strip()
                    
                    # Keep full title (including suffix like "– Cognizant Infra Services")
                    title = title_part.split('|')[0].strip()
                    
                    # Previous line is company
                    if i > 0:
                        prev_line = lines[i - 1].strip()
                        loc_match = re.search(r'[-–]\s*(.+)$', prev_line)
                        if loc_match:
                            employer = prev_line[:prev_line.find('-')].strip()
                            location = loc_match.group(1).strip()
                        else:
                            employer = prev_line
                else:
                    # Standard format: "Company – Title Location Date" or variations
                    header = line[:date_match.start()].strip()
                    
                    # Pattern 1: "Company – Title Location" with known location
                    comp_match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+?)\s+((?:Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]*(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)\s*$', header)
                    if comp_match:
                        employer = comp_match.group(1).strip()
                        title = comp_match.group(2).strip()
                        location = comp_match.group(3).strip()
                    else:
                        # Pattern 2: "Company – Title, Location Client – ClientName" (Sudheer TCS format)
                        # Also handles: "Company – Title"
                        dash_match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+)$', header)
                        if dash_match:
                            employer = dash_match.group(1).strip()
                            title_part = dash_match.group(2).strip()
                            
                            # Check if title_part contains "Client –" pattern
                            client_match = re.search(r'\s*Client\s*[-–]\s*(.+?)(?:\s*[-–]\s*(?:NA|N/A))?\s*$', title_part, re.IGNORECASE)
                            if client_match:
                                # Extract title before "Client"
                                title = title_part[:client_match.start()].strip().rstrip(',')
                                client = client_match.group(1).strip()
                            else:
                                title = title_part
                            
                            # Try to extract location from title if present
                            loc_match = re.search(r',?\s*(India|USA|Philadelphia|Chicago|Bangalore)\s*$', title, re.IGNORECASE)
                            if loc_match:
                                location = loc_match.group(1).strip()
                                title = title[:loc_match.start()].strip().rstrip(',')
                        else:
                            # Pattern 3: Company might be on this line, title on next
                            employer = header.strip()
                            if i + 1 < len(lines):
                                next_line = lines[i + 1].strip()
                                if not re.search(date_range_pattern, next_line) and not next_line.startswith(('•', '-', '*')):
                                    loc_match = re.search(r'((?:Philadelphia|Chicago|Bangalore|Hyderabad)[,\s]*(?:PA|TX|India|Karnataka)?)\s*$', next_line, re.IGNORECASE)
                                    if loc_match:
                                        title = next_line[:loc_match.start()].strip()
                                        location = loc_match.group(1).strip()
                                    else:
                                        title = next_line
                                    i += 1
                
                # Clean employer - remove title if accidentally included
                if not title and '–' in employer:
                    parts = employer.split('–', 1)
                    employer = parts[0].strip()
                    title = parts[1].strip() if len(parts) > 1 else ""
                
                # Get responsibilities
                responsibilities = []
                j = i + 1
                while j < len(lines) and j < i + 30:
                    resp_line = lines[j].strip()
                    
                    # Stop conditions
                    if re.search(date_range_pattern, resp_line, re.IGNORECASE):
                        break
                    if re.match(r'^(EDUCATION|TECHNICAL|SKILLS|CERTIFICATIONS|PERSONAL|KEY\s+ARCH|PROFESSIONAL\s+EXPERIENCE)', resp_line, re.IGNORECASE):
                        break
                    if re.match(r'^[A-Z][A-Za-z\s&]+[-–]\s*[A-Z]', resp_line):  # New company line
                        break
                    
                    # Extract bullet points
                    if resp_line.startswith(('•', '-', '*', '–')) or re.match(r'^\d+\.', resp_line):
                        resp_text = re.sub(r'^[•\-\*–\d.]\s*', '', resp_line)
                        if len(resp_text) > 20:
                            responsibilities.append(clean_output_text(resp_text))
                    
                    j += 1
                
                # Calculate duration
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                start_date = f"{start_year}-{(start_month or 1):02d}"
                end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
                
                # Extract tools from responsibilities
                tools = extract_tools_from_text(' '.join(responsibilities))
                
                if employer or title:
                    # Check if this is a duplicate
                    is_dup = any(
                        e.employer == employer and e.start_date == start_date
                        for e in experiences
                    )
                    if not is_dup:
                        experiences.append(ExperienceEntry(
                            employer=employer,
                            title=title.strip(),
                            location=location,
                            start_date=start_date,
                            end_date=end_date,
                            duration_months=duration,
                            responsibilities=responsibilities[:12],
                            tools=tools,
                            client=client
                        ))
        i += 1
    
    # Strategy 3: Detailed work experience section with company headers
    if not experiences or all(len(e.responsibilities) == 0 for e in experiences):
        experiences = extract_detailed_experiences(text, experiences)
    
    # Sort by start date descending
    experiences.sort(key=lambda x: x.start_date, reverse=True)
    
    return experiences


def extract_detailed_experiences(text: str, existing: List[ExperienceEntry]) -> List[ExperienceEntry]:
    """Extract from detailed WORK EXPERIENCE section and merge with existing jobs."""
    
    # Find work experience section
    work_match = re.search(r'WORK\s+EXPERIENCE[:\s]*\n(.+?)(?:\nPERSONAL|\nEDUCATION|\Z)', text, re.IGNORECASE | re.DOTALL)
    if not work_match:
        return existing
    
    work_section = work_match.group(1)
    lines = work_section.split('\n')
    
    # Parse into job blocks
    job_blocks = []
    current_block = {'company': '', 'title': '', 'client': '', 'responsibilities': []}
    in_responsibilities = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for company line (Title case company name)
        is_company = (
            re.match(r'^[A-Z][A-Za-z\s&.,()]+(?:Ltd|Inc|Corp|Technologies|Solutions|Pvt|Group|Mahindra|Capgemini|Synechron)?\.?$', line, re.IGNORECASE) and
            len(line) < 60 and
            '|' not in line and
            not line.lower().startswith(('key', 'project', 'designed', 'developed', 'implemented', 'automated', 'executed', 'validated', 'created', 'used', 'integrated', 'conducted', 'assisted', 'performed', 'monitored', 'ensured', 'participated', 'remove', 'record', 'master', 'understand'))
        )
        
        if is_company:
            # Save previous block if has data
            if current_block['company'] and current_block['responsibilities']:
                job_blocks.append(current_block.copy())
            
            current_block = {'company': line, 'title': '', 'client': '', 'responsibilities': []}
            in_responsibilities = False
            continue
        
        # Check for title/client line: "Title | Client: X | Project: Y"
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            current_block['title'] = parts[0]
            for part in parts:
                if 'Client' in part:
                    current_block['client'] = re.sub(r'Client[:\s]*', '', part).strip()
            in_responsibilities = False
            continue
        
        # Check for section headers
        if re.match(r'^(Key\s+Responsibilities|Project\s+Summary)[:\s]*$', line, re.IGNORECASE):
            in_responsibilities = 'Responsibilities' in line
            continue
        
        # Skip Project Summary content (usually one long line)
        if not in_responsibilities and 'Project Summary' not in line:
            if len(line) > 80 and current_block['title'] and not current_block['responsibilities']:
                continue
        
        # Collect responsibilities - handle BOTH bullet points AND plain text
        if in_responsibilities:
            resp = re.sub(r'^[•\-\*–]\s*', '', line)
            # Plain text lines that look like responsibilities (start with action verbs)
            if len(resp) > 25:
                current_block['responsibilities'].append(clean_output_text(resp))
        elif line.startswith(('•', '-', '*', '–')):
            resp = re.sub(r'^[•\-\*–]\s*', '', line)
            if len(resp) > 25:
                current_block['responsibilities'].append(clean_output_text(resp))
    
    # Don't forget last block
    if current_block['company'] and current_block['responsibilities']:
        job_blocks.append(current_block)
    
    # Debug: print what we found
    # for b in job_blocks:
    #     print(f"Block: {b['company']} | Client: {b['client']} | Resp: {len(b['responsibilities'])}")
    
    # Now match job_blocks with existing experiences
    for exp in existing:
        if exp.responsibilities:  # Already has responsibilities
            continue
        
        # Find matching block by employer/client name
        exp_emp_lower = exp.employer.lower().strip()
        best_match = None
        best_score = 0
        
        for block in job_blocks:
            block_company = block['company'].lower().strip()
            block_client = block['client'].lower().strip() if block['client'] else ''
            
            score = 0
            
            # Direct match with client (CME group == CME group)
            if block_client and (exp_emp_lower == block_client or exp_emp_lower in block_client or block_client in exp_emp_lower):
                score = 10
            # Direct match with company
            elif exp_emp_lower == block_company or exp_emp_lower in block_company or block_company in exp_emp_lower:
                score = 10
            # Partial word match
            else:
                exp_words = set(w for w in exp_emp_lower.split() if len(w) > 2)
                company_words = set(w for w in block_company.split() if len(w) > 2)
                client_words = set(w for w in block_client.split() if len(w) > 2) if block_client else set()
                
                # Check overlap
                company_overlap = len(exp_words & company_words)
                client_overlap = len(exp_words & client_words)
                
                score = max(company_overlap, client_overlap) * 2
            
            if score > best_score:
                best_score = score
                best_match = block
        
        if best_match and best_score >= 2:
            exp.responsibilities = best_match['responsibilities'][:12]
            exp.tools = extract_tools_from_text(' '.join(best_match['responsibilities']))
            if best_match['client']:
                exp.client = best_match['client']
    
    return existing


def extract_tools_from_text(text: str) -> List[str]:
    """Extract tools/technologies from text."""
    tools = set()
    text_lower = text.lower()
    
    tool_list = [
        # Cloud & Infra
        'aws', 'azure', 'gcp', 'ibm cloud', 'vmware vsphere', 'vcloud',
        'docker', 'kubernetes', 'terraform', 'ansible',
        # DevOps
        'jenkins', 'azure devops', 'github', 'gitlab', 'ci/cd',
        'power automate', 'automation scripting', 'scripting',
        # Databases
        'mongodb', 'cassandra', 'redis', 'postgresql', 'mysql', 'oracle', 'sql server',
        # Programming
        'python', 'java', 'javascript', 'sql', 'bash', 'powershell',
        # Testing
        'selenium', 'pytest', 'postman', 'cucumber',
        # PM Tools
        'jira', 'confluence', 'servicenow', 'ms project', 'clarity-ppm',
        # Visualization
        'prometheus', 'grafana', 'elk stack', 'tableau', 'power bi',
        # Frameworks
        'agile', 'waterfall', 'scrum', 'kanban', 'itil',
        # Other
        'linux', 'unix', 'git'
    ]
    
    for tool in tool_list:
        if re.search(rf'\b{re.escape(tool)}\b', text_lower):
            # Proper capitalization
            if tool in ['aws', 'gcp', 'sql', 'ci/cd']:
                tools.add(tool.upper())
            elif tool == 'azure devops':
                tools.add('Azure DevOps')
            elif tool == 'ibm cloud':
                tools.add('IBM Cloud')
            elif tool == 'vmware vsphere':
                tools.add('VMware vSphere')
            elif tool == 'power automate':
                tools.add('Power Automate')
            elif tool == 'power bi':
                tools.add('Power BI')
            elif tool == 'ms project':
                tools.add('MS Project')
            elif tool == 'clarity-ppm':
                tools.add('Clarity-PPM')
            elif tool == 'elk stack':
                tools.add('ELK Stack')
            elif tool == 'automation scripting':
                tools.add('Automation Scripting')
            else:
                tools.add(tool.title())
    
    return sorted(list(tools))


# ============================================================================
# EDUCATION EXTRACTION
# ============================================================================

def extract_education(text: str) -> List[Dict[str, str]]:
    """Extract education - handles multiple formats."""
    education = []
    
    # Find education section
    edu_match = re.search(
        r'EDUCATION(?:AL)?\s*(?:QUALIFICATION)?[:\s]*\n(.+?)(?:\nROLES|\nPROFESSIONAL|\nWORK|\nTECHNICAL|\nPERSONAL|\nCERTIFI|\nCORE|\nAS\s+A\s+SCRUM|\n_+|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if not edu_match:
        return education
    
    edu_section = edu_match.group(1)
    lines = [l.strip() for l in edu_section.split('\n') if l.strip() and not l.strip().startswith('_')]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip irrelevant lines
        if re.match(r'^(Worked|Working|•|-|\*|PROFESSIONAL|Roles|As\s+a\s+Scrum|Facilitate)', line, re.IGNORECASE):
            i += 1
            continue
        
        entry = {}
        
        # Format 1: "Degree | Institution | Month Year" (Sudheer/Pipe format)
        # Example: "Master of Science, Information Systems | Texas A&M International University | Dec 2015"
        if line.count('|') >= 2:
            parts = [p.strip() for p in line.split('|')]
            entry['degree'] = parts[0]
            entry['institution'] = parts[1]
            # Extract year from last part
            year_match = re.search(r'(\d{4})', parts[-1])
            if year_match:
                entry['year'] = year_match.group(1)
            education.append(entry)
            i += 1
            continue
        
        # Format 2: "Degree – Institution | Year-Year" (Jimmy format)
        # Example: "Master of Computer Applications (MCA) – IGNOU, New Delhi, India | 2001–2004"
        dash_pipe_match = re.match(r'^(.+?)\s*[-–]\s*(.+?)\s*\|\s*(\d{4})\s*[-–]\s*(\d{4})\s*$', line)
        if dash_pipe_match:
            entry['degree'] = dash_pipe_match.group(1).strip()
            entry['institution'] = dash_pipe_match.group(2).strip()
            entry['year'] = dash_pipe_match.group(4)  # End year
            education.append(entry)
            i += 1
            continue
        
        # Format 3: "Institution DateRange" then "Degree" on next line (Khaliq format)
        # Example: "University of Mysore Aug 2012 - Jun 2014" then "MBA, Marketing"
        date_range_match = re.search(r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4})', line)
        if date_range_match:
            inst_part = line[:date_range_match.start()].strip()
            year = date_range_match.group(2).split()[-1]  # Get end year
            
            if inst_part:
                entry['institution'] = inst_part
                entry['year'] = year
                
                # Next line should be degree
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not re.search(r'\d{4}\s*[-–]\s*\d{4}', next_line):
                        degree_keywords = ['master', 'bachelor', 'mba', 'mca', 'bca', 'b.tech', 'm.tech', 'engineering', 'science', 'arts']
                        if any(kw in next_line.lower() for kw in degree_keywords) or ',' in next_line:
                            entry['degree'] = next_line
                            i += 1
                
                education.append(entry)
                i += 1
                continue
        
        # Format 4: "Degree From Institution Year" (Madhuri format)
        from_match = re.match(r'^(.+?)\s+[Ff]rom\s+(.+?)\s+(?:\w+\s+)?(\d{4})$', line)
        if from_match:
            entry['degree'] = from_match.group(1).strip()
            entry['institution'] = from_match.group(2).strip()
            entry['year'] = from_match.group(3)
            education.append(entry)
            i += 1
            continue
        
        # Format 5: Simple degree line with institution
        degree_keywords = ['master', 'bachelor', 'mba', 'mca', 'bca', 'b.tech', 'm.tech', 'b.e', 'm.e', 'ph.d', 'me ', 'be ']
        if any(kw in line.lower() for kw in degree_keywords):
            parts = re.split(r'\s*[-–]\s*', line, maxsplit=1)
            entry['degree'] = parts[0].strip()
            
            if len(parts) > 1:
                rest = parts[1]
                year_match = re.search(r'(\d{4})\s*$', rest)
                if year_match:
                    entry['year'] = year_match.group(1)
                    entry['institution'] = rest[:year_match.start()].strip().rstrip(',|')
                else:
                    entry['institution'] = rest.strip()
            
            if entry.get('degree'):
                education.append(entry)
        
        i += 1
    
    # Ensure all entries have all fields (null if missing)
    cleaned = []
    for edu in education:
        cleaned.append({
            'degree': edu.get('degree') or None,
            'institution': edu.get('institution') or None,
            'year': edu.get('year') or None
        })
    
    return cleaned


# ============================================================================
# CERTIFICATION EXTRACTION
# ============================================================================

def extract_certifications(text: str) -> List[str]:
    """Extract certifications."""
    certifications = []
    
    # Multiple section patterns
    cert_patterns = [
        r'CERTIFICATIONS?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\nEDUCATION|\nSKILLS|\Z)',
        r'CERTIFICATES?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\Z)',
    ]
    
    cert_section = ""
    for pattern in cert_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            cert_section = match.group(1)
            break
    
    if cert_section:
        for line in cert_section.split('\n'):
            line = re.sub(r'^[•·\-\*]\s*', '', line.strip())
            if line and 3 < len(line) < 200:
                # Skip section headers and experience lines
                if re.match(r'^(PROFESSIONAL|EXPERIENCE|IMG|IBM|Cognizant|Dell|Wipro)', line, re.IGNORECASE):
                    continue
                
                # Clean "In Progress" suffix
                line = re.sub(r'\s*[-–]\s*In Progress\.?$', '', line, flags=re.IGNORECASE)
                
                # Split combined certifications on | 
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    for part in parts:
                        if part and len(part) > 3:
                            certifications.append(clean_output_text(part))
                else:
                    certifications.append(clean_output_text(line))
    
    return list(dict.fromkeys(certifications))[:20]


# ============================================================================
# SUMMARY EXTRACTION
# ============================================================================

def extract_summary(text: str) -> str:
    """Extract professional summary."""
    patterns = [
        r'(?:PROFESSIONAL\s+)?SUMMARY[:\s]*\n(.+?)(?:\nSKILLS|\nEXPERIENCE|\nWORK|\nTECHNICAL|\nCORE|\Z)',
        r'EXPERIENCE\s+SUMMARY[:\s]*\n(.+?)(?:\nTECHNICAL|\nSKILLS|\nPROFESSIONAL|\Z)',
        r'TECHNICAL\s+QUALIFICATION\s+HEADLINE[:\s]*\n(.+?)(?:\nEXPERIENCE|\nSKILLS|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            summary = match.group(1).strip()
            # Clean up
            summary = re.sub(r'\n+', ' ', summary)
            summary = clean_output_text(summary)
            return summary[:2000]
    
    return ""


# ============================================================================
# SKILL EXTRACTION WITH EXPERIENCE CALCULATION
# ============================================================================

def extract_technical_skills(text: str) -> List[str]:
    """Extract flat list of technical skills."""
    skills = set()
    text_lower = text.lower()
    
    for skill in ALL_SKILLS:
        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
            skills.add(skill.title() if len(skill) > 3 else skill.upper())
    
    return sorted(list(skills))


def calculate_key_skills(text: str, experiences: List[ExperienceEntry]) -> Dict[str, List[Dict]]:
    """Calculate skills with experience months - enterprise grade."""
    key_skills = {cat: [] for cat in SKILL_CATEGORIES}
    skill_months: Dict[str, int] = {}
    found_skills: Dict[str, str] = {}
    
    text_lower = text.lower()
    
    # Find all skills in resume
    for cat, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                display_name = skill.title() if len(skill) > 3 else skill.upper()
                found_skills[skill] = display_name
                skill_months[skill] = 0
    
    # Find which jobs mention which skills
    # We need to look at the FULL text around each job, not just extracted responsibilities
    
    # Build job sections from text
    job_sections = []
    for i, exp in enumerate(experiences):
        # Create searchable text for this job
        exp_text = f"{exp.title} {exp.employer} {' '.join(exp.responsibilities)} {' '.join(exp.tools)} {exp.client}".lower()
        job_sections.append((exp, exp_text))
    
    # For each skill, calculate total months
    for skill in found_skills:
        for exp, exp_text in job_sections:
            # Check if skill appears in this job's context
            if re.search(rf'\b{re.escape(skill)}\b', exp_text):
                skill_months[skill] = skill_months.get(skill, 0) + exp.duration_months
    
    # If no experience was calculated, try to assign based on overall resume mentions
    # and distribute across total experience
    total_months = sum(e.duration_months for e in experiences)
    for skill in found_skills:
        if skill_months.get(skill, 0) == 0 and total_months > 0:
            # Check if skill is in technical skills section or core competencies
            # If so, assume it was used across career
            skills_section = re.search(
                r'(?:TECHNICAL\s+SKILLS?|CORE\s+COMPETENCIES|SKILLS)[:\s]*\n(.+?)(?:\n[A-Z]{2,}|\Z)',
                text, re.IGNORECASE | re.DOTALL
            )
            if skills_section:
                section_text = skills_section.group(1).lower()
                if re.search(rf'\b{re.escape(skill)}\b', section_text):
                    # Skill is listed in skills section - assign total experience
                    skill_months[skill] = total_months
    
    # Build categorized output
    for cat, skills in SKILL_CATEGORIES.items():
        cat_skills = []
        for skill in skills:
            if skill in found_skills:
                cat_skills.append({
                    "skill": found_skills[skill],
                    "experience_months": skill_months.get(skill, 0)
                })
        # Sort by experience descending
        cat_skills.sort(key=lambda x: x['experience_months'], reverse=True)
        key_skills[cat] = cat_skills
    
    return key_skills


# ============================================================================
# AGENTIC FRAMEWORK - MULTI-AGENT PARSING SYSTEM
# ============================================================================
"""
AGENTIC ARCHITECTURE:
=====================
1. PARSER AGENT (Primary) - Regex-based multi-pattern extraction
2. VALIDATION AGENT - Quality checks & issue detection  
3. AI FALLBACK AGENT - Claude API for failed/incomplete parses

Flow: Parser → Validation → (if issues) AI Fallback → Final Result
"""

class ValidationResult:
    """Result of validation agent checks."""
    def __init__(self):
        self.issues = []
        self.score = 100
        self.needs_ai_fallback = False
        self.missing_fields = []


def validation_agent(parsed: Dict, text: str) -> ValidationResult:
    """
    VALIDATION AGENT: Check parsed result quality and identify issues.
    Returns ValidationResult with score and issues list.
    """
    result = ValidationResult()
    pr = parsed.get("parsed_resume", {})
    
    # Critical field checks (20 points each)
    critical_checks = [
        ("name", pr.get("name") and len(pr.get("name", "")) > 2 and " " in pr.get("name", "")),
        ("email", bool(pr.get("email") and "@" in pr.get("email", ""))),
        ("experience", len(pr.get("experience", [])) > 0),
    ]
    
    for field, passed in critical_checks:
        if not passed:
            result.issues.append(f"missing_{field}")
            result.missing_fields.append(field)
            result.score -= 20
    
    # Important field checks (10 points each)
    important_checks = [
        ("phone", bool(pr.get("phone_number"))),
        ("title", bool(pr.get("title"))),
        ("education", len(pr.get("education", [])) > 0),
    ]
    
    for field, passed in important_checks:
        if not passed:
            result.issues.append(f"missing_{field}")
            result.missing_fields.append(field)
            result.score -= 10
    
    # Experience quality checks
    experiences = pr.get("experience", [])
    if experiences:
        total_resp = sum(len(e.get("responsibilities", [])) for e in experiences)
        if total_resp < 5:
            result.issues.append("low_responsibilities")
            result.missing_fields.append("responsibilities")
            result.score -= 15
        
        # Check for missing titles/employers
        for i, exp in enumerate(experiences):
            if not exp.get("title"):
                result.issues.append(f"exp_{i}_missing_title")
                result.score -= 5
            if not exp.get("Employer"):
                result.issues.append(f"exp_{i}_missing_employer")
                result.score -= 5
    
    # Education quality checks
    for edu in pr.get("education", []):
        if not edu.get("degree"):
            result.issues.append("edu_missing_degree")
            result.score -= 5
    
    # Name sanity check - should not be tech terms
    name = pr.get("name", "").lower()
    invalid_names = ["sql server", "data engineer", "software engineer", "project manager", 
                     "key expertise", "additional details", "professional", "reporting tools",
                     "results-driven", "experienced", "senior"]
    if any(inv in name for inv in invalid_names):
        result.issues.append("invalid_name")
        result.missing_fields.append("name")
        result.score -= 25
    
    # Determine if AI fallback needed
    result.needs_ai_fallback = result.score < 60 or "missing_name" in result.issues or "invalid_name" in result.issues
    
    return result


async def ai_fallback_agent(text: str, parsed: Dict, validation: ValidationResult) -> Dict:
    """
    AI FALLBACK AGENT: Use Claude API to extract/fix missing fields.
    Only called when validation score is low or critical fields missing.
    """
    if not ANTHROPIC_API_KEY:
        parsed["ai_skipped"] = "No API key"
        return parsed
    
    try:
        import httpx
        
        # Build targeted prompt based on what's missing
        missing = list(set(validation.missing_fields))
        
        prompt = f"""You are a resume parsing expert. Extract the following MISSING fields from this resume.
Return ONLY valid JSON with the requested fields, no markdown, no explanation.

MISSING FIELDS TO EXTRACT: {', '.join(missing)}

RESUME TEXT:
{text[:15000]}

Return JSON in this exact format (only include fields that are missing):
{{
  "firstname": "First name only",
  "lastname": "Last name only", 
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1234567890",
  "title": "Professional Title like 'Data Engineer' or 'Project Manager'",
  "location": "City, State",
  "education": [
    {{"degree": "Masters in Computer Science", "institution": "University Name", "year": "2020"}}
  ],
  "experience": [
    {{
      "Employer": "Company Name",
      "title": "Job Title",
      "location": "City, State",
      "start_date": "2020-01",
      "end_date": "2023-12",
      "duration_months": 48,
      "responsibilities": ["Responsibility 1", "Responsibility 2"],
      "tools": ["Python", "SQL"]
    }}
  ]
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
                ai_response = response.json()
                ai_text = ai_response.get("content", [{}])[0].get("text", "")
                
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', ai_text)
                if json_match:
                    ai_data = json.loads(json_match.group())
                    pr = parsed.get("parsed_resume", {})
                    
                    # Merge AI results with existing parsed data
                    if "name" in missing or "invalid_name" in validation.issues:
                        if ai_data.get("firstname"):
                            pr["firstname"] = ai_data["firstname"]
                        if ai_data.get("lastname"):
                            pr["lastname"] = ai_data["lastname"]
                        if ai_data.get("name"):
                            pr["name"] = ai_data["name"]
                    
                    if "email" in missing and ai_data.get("email"):
                        pr["email"] = ai_data["email"]
                    
                    if "phone" in missing and ai_data.get("phone"):
                        pr["phone_number"] = ai_data["phone"]
                    
                    if "title" in missing and ai_data.get("title"):
                        pr["title"] = ai_data["title"]
                    
                    if "education" in missing and ai_data.get("education"):
                        pr["education"] = ai_data["education"]
                    
                    if "experience" in missing and ai_data.get("experience"):
                        # Merge or replace experience
                        if not pr.get("experience") or len(pr["experience"]) == 0:
                            pr["experience"] = ai_data["experience"]
                        else:
                            # Add responsibilities to existing experiences if missing
                            for ai_exp in ai_data.get("experience", []):
                                for existing_exp in pr["experience"]:
                                    if not existing_exp.get("responsibilities") or len(existing_exp.get("responsibilities", [])) == 0:
                                        existing_exp["responsibilities"] = ai_exp.get("responsibilities", [])
                                        existing_exp["tools"] = ai_exp.get("tools", [])
                                        break
                    
                    if "responsibilities" in missing and ai_data.get("experience"):
                        # Just add responsibilities to existing experiences
                        ai_exps = ai_data.get("experience", [])
                        for i, existing_exp in enumerate(pr.get("experience", [])):
                            if not existing_exp.get("responsibilities") or len(existing_exp.get("responsibilities", [])) == 0:
                                if i < len(ai_exps):
                                    existing_exp["responsibilities"] = ai_exps[i].get("responsibilities", [])
                                    existing_exp["tools"] = ai_exps[i].get("tools", [])
                    
                    # Recalculate totals
                    if pr.get("experience"):
                        pr["total_experience_months"] = sum(e.get("duration_months", 0) for e in pr["experience"])
                        pr["total_experience_years"] = round(pr["total_experience_months"] / 12, 1) if pr["total_experience_months"] else 0
                    
                    parsed["ai_enhanced"] = True
                    parsed["ai_fields_fixed"] = missing
                    parsed["validation_score_before"] = validation.score
            else:
                parsed["ai_error"] = f"API returned {response.status_code}"
                    
    except Exception as e:
        parsed["ai_error"] = str(e)
    
    return parsed


def validate_and_fix_result(result: Dict, text: str) -> Dict:
    """Validate parsed result and apply fallback fixes for any missing data."""
    pr = result.get("parsed_resume", {})
    
    # Track what was fixed
    fixes_applied = []
    
    # 1. Validate name - must have at least firstname or lastname
    if not pr.get("firstname") and not pr.get("lastname"):
        # Fallback: try to extract from first non-header line
        lines = [l.strip() for l in text.split('\n') if l.strip()][:10]
        for line in lines:
            if line.lower() not in ['resume', 'cv', 'curriculum vitae']:
                parts = line.split()
                if 2 <= len(parts) <= 4:
                    pr["firstname"] = parts[0]
                    pr["lastname"] = parts[-1]
                    pr["name"] = line
                    fixes_applied.append("name_fallback")
                    break
    
    # 2. Validate title - should not be empty for professional resumes
    if not pr.get("title") and pr.get("experience"):
        # Fallback: use first job title
        pr["title"] = pr["experience"][0].get("title")
        fixes_applied.append("title_from_experience")
    
    # 3. Validate experience - each entry must have Employer and title
    for exp in pr.get("experience", []):
        if not exp.get("Employer") and exp.get("title"):
            # Sometimes title contains employer
            if " at " in exp["title"].lower() or " @ " in exp["title"]:
                parts = re.split(r'\s+(?:at|@)\s+', exp["title"], flags=re.IGNORECASE)
                if len(parts) == 2:
                    exp["title"] = parts[0]
                    exp["Employer"] = parts[1]
                    fixes_applied.append("employer_from_title")
        
        if not exp.get("title") and exp.get("Employer"):
            # Title might be in responsibilities header
            exp["title"] = "Professional"  # Generic fallback
            fixes_applied.append("generic_title")
    
    # 4. Validate education - ensure proper structure
    for edu in pr.get("education", []):
        if not edu.get("degree") and edu.get("institution"):
            # Sometimes institution contains degree info
            inst = edu["institution"]
            if any(kw in inst.lower() for kw in ["master", "bachelor", "mba", "mca"]):
                edu["degree"] = inst
                edu["institution"] = None
                fixes_applied.append("degree_from_institution")
    
    # 5. Validate contact info
    if not pr.get("email"):
        # Try harder to find email
        email_match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text)
        if email_match:
            pr["email"] = email_match.group(1).lower()
            fixes_applied.append("email_fallback")
    
    if not pr.get("phone_number"):
        # Try harder to find phone
        phone_match = re.search(r'(\+?[\d\s\-().]{10,})', text)
        if phone_match:
            phone = re.sub(r'[^\d+\-() ]', '', phone_match.group(1))
            if len(re.sub(r'\D', '', phone)) >= 10:
                pr["phone_number"] = phone.strip()
                fixes_applied.append("phone_fallback")
    
    # 6. Calculate total experience if not set
    if pr.get("experience") and pr.get("total_experience_months", 0) == 0:
        pr["total_experience_months"] = sum(e.get("duration_months", 0) for e in pr["experience"])
        pr["total_experience_years"] = round(pr["total_experience_months"] / 12, 1)
        fixes_applied.append("experience_calc")
    
    # Add validation metadata
    if fixes_applied:
        result["validation_fixes"] = fixes_applied
    
    return result


def ensure_data_quality(experiences: List[ExperienceEntry]) -> List[ExperienceEntry]:
    """Ensure each experience entry has minimum required data."""
    quality_experiences = []
    
    for exp in experiences:
        # Skip entries with no employer AND no title
        if not exp.employer and not exp.title:
            continue
        
        # Ensure duration is positive
        if exp.duration_months <= 0:
            exp.duration_months = 1
        
        # Clean up empty strings to None
        if exp.employer == "":
            exp.employer = "Unknown Company"
        if exp.title == "":
            exp.title = "Professional"
        
        quality_experiences.append(exp)
    
    return quality_experiences

async def validate_with_ai(resume_text: str, parsed: Dict) -> Dict:
    """Use Claude API to validate and enhance parsed results."""
    if not ANTHROPIC_API_KEY:
        return parsed
    
    pr = parsed.get('parsed_resume', {})
    
    # Check for issues
    issues = []
    if not pr.get('firstname') or not pr.get('lastname'):
        issues.append("name")
    if len(pr.get('experience', [])) < 1:
        issues.append("experience")
    if len(pr.get('education', [])) == 0:
        issues.append("education")
    if not pr.get('title'):
        issues.append("title")
    
    if not issues:
        return parsed
    
    try:
        import httpx
        
        prompt = f"""Extract these missing fields from the resume. Return ONLY valid JSON.

Missing: {', '.join(issues)}

Resume:
{resume_text[:10000]}

Return JSON:
{{
  "firstname": "...",
  "lastname": "...",
  "title": "Professional title",
  "education": [{{"degree": "...", "institution": "...", "year": "YYYY"}}],
  "experience": [{{"Employer": "...", "title": "...", "start_date": "YYYY-MM", "end_date": "YYYY-MM", "duration_months": N}}]
}}"""

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 4000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('content', [{}])[0].get('text', '{}')
                
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    ai_data = json.loads(json_match.group())
                    
                    # Merge results
                    for key in ['firstname', 'lastname', 'title']:
                        if ai_data.get(key) and not pr.get(key):
                            pr[key] = ai_data[key]
                    
                    if ai_data.get('education') and not pr.get('education'):
                        pr['education'] = ai_data['education']
                    
                    if ai_data.get('experience') and len(ai_data['experience']) > len(pr.get('experience', [])):
                        pr['experience'] = ai_data['experience']
                    
                    if pr.get('firstname') and pr.get('lastname'):
                        pr['name'] = f"{pr['firstname']} {pr['lastname']}"
                    
                    parsed['ai_enhanced'] = True
                    parsed['ai_issues_fixed'] = issues
    
    except Exception as e:
        parsed['ai_error'] = str(e)
    
    return parsed


# ============================================================================
# MAIN PARSING FUNCTION
# ============================================================================

async def parse_resume_full(params: ParseResumeInput) -> str:
    """Parse resume and return structured JSON - enterprise grade with validation."""
    text = normalize_text(params.resume_text)
    
    # Extract all components
    contact = extract_contact(text)
    firstname, middle, lastname = extract_name(text)
    experiences = extract_experiences(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    summary = extract_summary(text)
    title = extract_title(text, experiences)
    
    # Apply data quality checks on experiences
    experiences = ensure_data_quality(experiences)
    
    # Build name
    name_parts = [firstname]
    if middle:
        name_parts.append(middle)
    name_parts.append(lastname)
    name = ' '.join(filter(None, name_parts))
    
    # Get location - prefer most recent job, fallback to contact
    location = ""
    if experiences and experiences[0].location:
        location = experiences[0].location
    if not location:
        location = contact.get('location', '')
    
    # Calculate skills with experience
    key_skills = calculate_key_skills(text, experiences)
    
    # Build result
    result = {
        "parsed_resume": {
            "firstname": firstname if firstname else None,
            "lastname": lastname if lastname else None,
            "name": name if name else None,
            "title": title if title else None,
            "location": location if location else None,
            "phone_number": contact.get('phone') if contact.get('phone') else None,
            "email": contact.get('email') if contact.get('email') else None,
            "linkedin": contact.get('linkedin') if contact.get('linkedin') else None,
            "summary": summary if summary else None,
            "total_experience_months": sum(e.duration_months for e in experiences),
            "total_experience_years": round(sum(e.duration_months for e in experiences) / 12, 1) if experiences else 0,
            "technical_skills": extract_technical_skills(text),
            "key_skills": key_skills,
            "education": education,
            "certifications": certifications,
            "experience": [
                {
                    "Employer": clean_output_text(e.employer) if e.employer else None,
                    "title": clean_output_text(e.title) if e.title else None,
                    "location": clean_output_text(e.location) if e.location else None,
                    "start_date": e.start_date,
                    "end_date": e.end_date,
                    "duration_months": e.duration_months,
                    "responsibilities": [clean_output_text(r) for r in e.responsibilities],
                    "tools": e.tools
                }
                for e in experiences
            ],
            "filename": params.filename or ""
        }
    }
    
    # =========================================================================
    # AGENTIC FRAMEWORK EXECUTION
    # =========================================================================
    
    # Step 1: Apply basic validation fixes
    result = validate_and_fix_result(result, text)
    
    # Step 2: Run Validation Agent to check quality
    validation = validation_agent(result, text)
    result["validation_score"] = validation.score
    result["validation_issues"] = validation.issues
    
    # Step 3: If validation fails, use AI Fallback Agent
    if validation.needs_ai_fallback and (params.use_ai_validation or validation.score < 50):
        result = await ai_fallback_agent(text, result, validation)
    elif params.use_ai_validation and ANTHROPIC_API_KEY:
        # Legacy AI validation for backward compatibility
        result = await validate_with_ai(text, result)
    
    return json.dumps(result, indent=2, ensure_ascii=False)


# ============================================================================
# MCP TOOLS
# ============================================================================

if MCP_AVAILABLE and mcp:
    @mcp.tool(name="resume_parse_full")
    async def resume_parse_full_tool(params: ParseResumeInput) -> str:
        return await parse_resume_full(params)


if __name__ == "__main__":
    if MCP_AVAILABLE and mcp:
        mcp.run()
