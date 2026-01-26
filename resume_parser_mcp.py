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
        "angular", "vue", "django", "flask", ".net", "asp.net", "vb.net", "gherkin"
    ],
    "Databases": [
        "mongodb", "cassandra", "redis", "postgresql", "mysql", "oracle", "db2",
        "sql server", "nosql", "dynamodb", "elasticsearch", "neo4j", "sql developer"
    ],
    "Cloud": [
        "aws", "azure", "gcp", "google cloud", "ibm cloud", "openstack",
        "ec2", "s3", "lambda", "ecs", "ecr", "eks", "rds", "vpc", "route 53",
        "multi-cloud", "hybrid cloud", "cloud computing"
    ],
    "DevOps": [
        "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
        "jenkins", "gitlab", "github actions", "ci/cd", "cicd", "concourse",
        "helm", "gitops", "devops", "devsecops", "infrastructure as code"
    ],
    "Testing & QA": [
        "selenium", "pytest", "unittest", "testng", "junit", "cucumber", "behave",
        "bdd", "tdd", "manual testing", "automation testing", "regression testing",
        "api testing", "postman", "functional testing", "uat", "quality assurance"
    ],
    "Security": [
        "sonarqube", "qualys", "crowdstrike", "mend", "uptycs", "snyk",
        "siem", "splunk", "palo alto", "firewall", "iam", "pam", "mfa",
        "cisco security", "checkpoint", "fortinet", "zscaler"
    ],
    "Data Science & Visualization": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "pandas", "numpy", "tableau", "power bi", "grafana", "prometheus",
        "elk stack", "kibana", "data visualization", "bi reporting"
    ],
    "Other Tools": [
        "jira", "confluence", "servicenow", "git", "github", "agile", "scrum",
        "kanban", "waterfall", "itil", "pmp", "project management", "linux",
        "unix", "rhel", "ubuntu", "centos", "windows server", "vmware", "hyper-v",
        "putty", "pycharm", "hp alm", "zephyr"
    ],
    "Business Domains": [
        "banking", "finance", "financial services", "telecom", "telecommunications",
        "healthcare", "pharma", "insurance", "retail", "e-commerce", "crm",
        "bss", "oss", "billing", "payment"
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


def clean_output_text(text: str) -> str:
    """Clean text for final output - remove artifacts."""
    if not text:
        return ""
    # Remove literal \n
    text = text.replace('\\n', ' ')
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
    lines = [l.strip() for l in text.split('\n') if l.strip()][:15]
    
    for line in lines:
        # Skip headers
        if line.lower() in ['resume', 'cv', 'curriculum vitae']:
            continue
        if re.search(r'@|linkedin|phone|email|mob:|\d{3}[-.\s]\d{3}', line, re.IGNORECASE):
            continue
        if re.match(r'^(Summary|Skills|Experience|Education|Core|Technical)', line, re.IGNORECASE):
            continue
        
        # Clean line
        name = re.sub(r'\s+', ' ', line)
        name = re.sub(r'[\|,].*$', '', name).strip()
        
        parts = name.split()
        if 2 <= len(parts) <= 4 and all(p[0].isupper() for p in parts if p):
            if len(parts) == 2:
                return parts[0], "", parts[1]
            elif len(parts) == 3:
                return parts[0], parts[1], parts[2]
            else:
                return parts[0], ' '.join(parts[1:-1]), parts[-1]
    
    return "", "", ""


def extract_title(text: str, experiences: List[ExperienceEntry]) -> str:
    """Extract professional title from summary or experience."""
    
    # Try from most recent job title first (most reliable)
    if experiences:
        title = experiences[0].title
        if title and len(title) > 3:
            # Remove location from title
            title = re.sub(r'\s+(Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago|India|USA|Karnataka)[,\s]*(?:PA|TX|IL|NY|CA|USA|India)?$', '', title, flags=re.IGNORECASE)
            return clean_output_text(title)
    
    # Try from summary patterns
    summary_patterns = [
        r'(\d+\+?\s+[Yy]ears?\s+(?:of\s+)?experience\s+(?:in|as)\s+)([\w\s]+?)(?:\.|including|,)',
        r'^([\w\s]+(?:Engineer|Developer|Manager|Analyst|Consultant|Lead|Architect|Specialist|Tester))',
    ]
    
    # Find summary section
    summary_match = re.search(r'(?:EXPERIENCE\s+)?SUMMARY[:\s]*\n(.+?)(?:\n[A-Z]{2,}|\Z)', text, re.IGNORECASE | re.DOTALL)
    if summary_match:
        summary = summary_match.group(1)
        for pattern in summary_patterns:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                # Get the role part
                if match.lastindex and match.lastindex >= 2:
                    role = match.group(2).strip()
                else:
                    role = match.group(1).strip()
                # Clean it up
                role = re.sub(r'^(PROFESSIONAL\s+SUMMARY\s*)', '', role, flags=re.IGNORECASE)
                if 'Engineer' in role or 'Manager' in role or 'Consultant' in role or 'Developer' in role or 'Analyst' in role or 'Tester' in role:
                    return clean_output_text(role[:60])
    
    return ""


# ============================================================================
# DATE PARSING
# ============================================================================

def parse_date(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse date string to (year, month, is_present)."""
    if not text:
        return None, None, False
    
    text = text.strip().lower()
    
    if any(p in text for p in ['present', 'current', 'now', 'ongoing', 'till date']):
        now = datetime.now()
        return now.year, now.month, True
    
    # Month Year
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*[,.]?\s*(\d{4})', text)
    if match:
        month = MONTH_MAP.get(match.group(1)[:3])
        year = int(match.group(2))
        return year, month, False
    
    # Just year
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if match:
        return int(match.group(1)), None, False
    
    return None, None, False


def calculate_duration(start_year: int, start_month: int, end_year: int, end_month: int) -> int:
    """Calculate duration in months."""
    start_dt = datetime(start_year, start_month or 1, 1)
    end_dt = datetime(end_year, end_month or 12, 1)
    delta = relativedelta(end_dt, start_dt)
    return max(1, delta.years * 12 + delta.months)


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
    
    # Strategy 2: Line-by-line date range detection
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
                    
                    # Clean title
                    title = re.sub(r'\s*[-–]\s*.*$', '', title_part.split('|')[0]).strip()
                    
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
                    # Standard format: "Company – Title Location Date"
                    header = line[:date_match.start()].strip()
                    
                    comp_match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+?)\s+((?:Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]*(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)\s*$', header)
                    if comp_match:
                        employer = comp_match.group(1).strip()
                        title = comp_match.group(2).strip()
                        location = comp_match.group(3).strip()
                    else:
                        # Company might be on this line, title on next
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
                
                # Clean employer
                employer = re.sub(r'\s*[-–].*$', '', employer).strip()
                
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
                    # Clean title - remove location if appended
                    title = re.sub(r'\s+(Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]*(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?$', '', title, flags=re.IGNORECASE)
                    
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
    """Extract from detailed WORK EXPERIENCE section with project info."""
    
    # Find work experience section
    work_match = re.search(r'WORK\s+EXPERIENCE[:\s]*\n(.+?)(?:\nPERSONAL|\nEDUCATION|\Z)', text, re.IGNORECASE | re.DOTALL)
    if not work_match:
        return existing
    
    work_section = work_match.group(1)
    
    # Pattern for company blocks
    # e.g., "Tech Mahindra" or "Synechron Technologies Pvt. Ltd."
    company_pattern = r'^([A-Z][A-Za-z\s&.,()]+(?:Ltd|Inc|Corp|Technologies|Solutions|Pvt|Group)?\.?)\s*$'
    
    lines = work_section.split('\n')
    current_company = ""
    current_title = ""
    current_client = ""
    current_responsibilities = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check for company line
        if re.match(company_pattern, line, re.IGNORECASE):
            current_company = line
            continue
        
        # Check for title/client line: "Title | Client: X | Project: Y"
        if '|' in line and 'Client' in line:
            parts = line.split('|')
            current_title = parts[0].strip()
            for part in parts[1:]:
                if 'Client' in part:
                    current_client = re.sub(r'Client[:\s]*', '', part).strip()
            continue
        
        # Check for "Key Responsibilities:" header
        if re.match(r'^Key\s+Responsibilities', line, re.IGNORECASE):
            continue
        
        # Bullet points are responsibilities
        if line.startswith(('•', '-', '*')) or re.match(r'^\d+\.', line):
            resp = re.sub(r'^[•\-\*\d.]\s*', '', line)
            if len(resp) > 20:
                current_responsibilities.append(clean_output_text(resp))
    
    # Update existing experiences with responsibilities
    for exp in existing:
        if not exp.responsibilities:
            # Find matching responsibilities
            for resp in current_responsibilities:
                if any(tool.lower() in resp.lower() for tool in exp.tools) or len(exp.responsibilities) < 5:
                    if resp not in exp.responsibilities:
                        exp.responsibilities.append(resp)
    
    return existing


def extract_tools_from_text(text: str) -> List[str]:
    """Extract tools/technologies from text."""
    tools = set()
    text_lower = text.lower()
    
    tool_list = [
        'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'gcp',
        'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
        'mongodb', 'cassandra', 'redis', 'postgresql', 'mysql', 'oracle',
        'kafka', 'spark', 'airflow', 'prometheus', 'grafana', 'elk stack',
        'react', 'angular', 'node.js', 'linux', 'jira', 'servicenow',
        'selenium', 'pytest', 'postman', 'git', 'tableau', 'power bi',
        'informatica', 'cucumber', 'behave', 'pycharm', 'unix'
    ]
    
    for tool in tool_list:
        if re.search(rf'\b{re.escape(tool)}\b', text_lower):
            tools.add(tool.title() if len(tool) > 3 else tool.upper())
    
    return sorted(list(tools))


# ============================================================================
# EDUCATION EXTRACTION
# ============================================================================

def extract_education(text: str) -> List[Dict[str, str]]:
    """Extract education - handles multiple formats."""
    education = []
    
    # Find education section
    edu_match = re.search(
        r'EDUCATION(?:AL)?\s*(?:QUALIFICATION)?[:\s]*\n(.+?)(?:\nROLES|\nPROFESSIONAL|\nWORK|\nTECHNICAL|\nPERSONAL|\nCERTIFI|\nCORE|\nAS\s+A\s+SCRUM|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if not edu_match:
        return education
    
    edu_section = edu_match.group(1)
    
    for line in edu_section.split('\n'):
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        # Skip irrelevant lines
        if re.match(r'^(Worked|Working|•|-|\*|PROFESSIONAL|Roles|As\s+a\s+Scrum|Facilitate|Remove|Coach|Shield|Foster)', line, re.IGNORECASE):
            continue
        
        # Skip lines that look like responsibilities
        if 'responsibilities' in line.lower() or 'impediments' in line.lower():
            continue
        
        entry = {}
        
        # Format 1: "Degree From Institution Year"
        from_match = re.match(r'^(.+?)\s+[Ff]rom\s+(.+?)\s+(?:\w+\s+)?(\d{4})$', line)
        if from_match:
            entry['degree'] = from_match.group(1).strip()
            entry['institution'] = from_match.group(2).strip()
            entry['year'] = from_match.group(3)
            education.append(entry)
            continue
        
        # Format 2: "Degree | Institution | Year" or with year range
        pipe_match = re.match(r'^(.+?)\s*\|\s*(.+?)$', line)
        if pipe_match:
            degree_part = pipe_match.group(1).strip()
            rest = pipe_match.group(2).strip()
            
            # Skip if degree part doesn't look like education
            if not any(kw in degree_part.lower() for kw in ['master', 'bachelor', 'mba', 'mca', 'bca', 'b.tech', 'm.tech', 'b.e', 'm.e', 'ph.d', 'degree', 'science', 'engineering', 'arts']):
                continue
            
            # Check if rest contains year range
            year_range_match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', rest)
            if year_range_match:
                entry['degree'] = degree_part
                entry['year'] = year_range_match.group(2)
                
                # Try to extract institution from degree part
                inst_match = re.search(r'[-–]\s*(.+)$', degree_part)
                if inst_match:
                    entry['institution'] = inst_match.group(1).strip()
                    entry['degree'] = degree_part[:degree_part.find('-')].strip()
            else:
                entry['degree'] = degree_part
                entry['institution'] = rest
                year_match = re.search(r'(\d{4})', line)
                if year_match:
                    entry['year'] = year_match.group(1)
            
            if entry.get('degree'):
                education.append(entry)
            continue
        
        # Format 3: Institution name with date range (split across lines)
        inst_match = re.match(r'^(University\s+of\s+\w+|[\w\s]+University|JNT\s+University)', line, re.IGNORECASE)
        if inst_match:
            entry['institution'] = inst_match.group(1).strip()
            year_match = re.search(r'(\d{4})', line)
            if year_match:
                entry['year'] = year_match.group(1)
            # Degree will be on next line - we'll handle this case separately
            education.append(entry)
            continue
        
        # Format 4: Degree keywords on single line
        degree_keywords = ['master', 'bachelor', 'mba', 'mca', 'bca', 'b.tech', 'm.tech', 'b.e', 'm.e', 'ph.d', 'me ', 'be ']
        if any(kw in line.lower() for kw in degree_keywords):
            # Try to parse "Degree - Institution"
            parts = re.split(r'\s*[-–]\s*', line, maxsplit=1)
            entry['degree'] = parts[0].strip()
            
            if len(parts) > 1:
                rest = parts[1]
                year_match = re.search(r'[|,]\s*(?:\d{4}\s*[-–]\s*)?(\d{4})\s*$', rest)
                if year_match:
                    entry['year'] = year_match.group(1)
                    entry['institution'] = rest[:year_match.start()].strip().rstrip(',|')
                else:
                    entry['institution'] = rest
                    year_match = re.search(r'(\d{4})', line)
                    if year_match:
                        entry['year'] = year_match.group(1)
            else:
                for pattern in [r'(University[\w\s,]+)', r'([\w\s]+University)', r'(IGNOU|Pondicherry|Texas A&M|Pune)']:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        entry['institution'] = match.group(1).strip()
                        break
                year_match = re.search(r'(\d{4})', line)
                if year_match:
                    entry['year'] = year_match.group(1)
            
            if entry.get('degree'):
                education.append(entry)
    
    # Clean up - merge institution-only entries with degree-only entries
    cleaned = []
    for i, edu in enumerate(education):
        if edu.get('degree') and not edu.get('institution'):
            # Check if next entry has institution only
            if i + 1 < len(education) and education[i+1].get('institution') and not education[i+1].get('degree'):
                edu['institution'] = education[i+1]['institution']
                if education[i+1].get('year'):
                    edu['year'] = education[i+1]['year']
                cleaned.append(edu)
                education[i+1] = {}  # Mark as used
            else:
                cleaned.append(edu)
        elif edu.get('institution') or edu.get('degree'):
            if edu:  # Not marked as used
                cleaned.append(edu)
    
    return [e for e in cleaned if e]


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
# CLAUDE AI VALIDATION
# ============================================================================

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
    """Parse resume and return structured JSON - enterprise grade."""
    text = normalize_text(params.resume_text)
    
    # Extract all components
    contact = extract_contact(text)
    firstname, middle, lastname = extract_name(text)
    experiences = extract_experiences(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    summary = extract_summary(text)
    title = extract_title(text, experiences)
    
    # Build name
    name_parts = [firstname]
    if middle:
        name_parts.append(middle)
    name_parts.append(lastname)
    name = ' '.join(filter(None, name_parts))
    
    # Calculate skills with experience
    key_skills = calculate_key_skills(text, experiences)
    
    # Build result
    result = {
        "parsed_resume": {
            "firstname": firstname,
            "lastname": lastname,
            "name": name,
            "title": title,
            "location": contact.get('location', ''),
            "phone_number": contact.get('phone', ''),
            "email": contact.get('email', ''),
            "linkedin": contact.get('linkedin', ''),
            "summary": summary,
            "total_experience_months": sum(e.duration_months for e in experiences),
            "total_experience_years": round(sum(e.duration_months for e in experiences) / 12, 1),
            "technical_skills": extract_technical_skills(text),
            "key_skills": key_skills,
            "education": education,
            "certifications": certifications,
            "experience": [
                {
                    "Employer": clean_output_text(e.employer),
                    "title": clean_output_text(e.title),
                    "location": clean_output_text(e.location),
                    "client": clean_output_text(e.client) if e.client else "",
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
    
    # AI validation if enabled and needed
    if params.use_ai_validation and ANTHROPIC_API_KEY:
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
