"""
Resume Parser MCP Server - Google-Level Production Grade
=========================================================
Enterprise resume parsing with Claude AI validation for accuracy.
Supports PDF, Word documents with intelligent multi-format extraction.

Configuration:
- Set ANTHROPIC_API_KEY environment variable for AI validation
- AI validation auto-activates when critical fields are missing
"""

import json
import re
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict, field_validator
import unicodedata

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("resume_parser_mcp")
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp = None


# ============================================================================
# CONFIGURATION - Set your API key here or via environment variable
# ============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # <-- ADD YOUR KEY HERE


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
        "apache kafka", "kafka", "snowflake", "databricks", "dbt", "glue", "redshift"
    ],
    "Programming": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "cobol", "bash", "shell",
        "powershell", "sql", "html", "css", "node.js", "nodejs", "react", "reactjs",
        "angular", "vue", "django", "flask", ".net", "asp.net", "vb.net"
    ],
    "Databases": [
        "mongodb", "cassandra", "redis", "postgresql", "mysql", "oracle", "db2",
        "sql server", "nosql", "dynamodb", "elasticsearch", "neo4j"
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
    "Security": [
        "sonarqube", "qualys", "crowdstrike", "mend", "uptycs", "snyk",
        "siem", "splunk", "palo alto", "firewall", "iam", "pam", "mfa",
        "cisco security", "checkpoint", "fortinet", "zscaler"
    ],
    "Data Science & Visualization": [
        "machine learning", "deep learning", "tensorflow", "pytorch",
        "pandas", "numpy", "tableau", "power bi", "grafana", "prometheus",
        "elk stack", "kibana", "data visualization"
    ],
    "Other Tools": [
        "jira", "confluence", "servicenow", "git", "agile", "scrum", "kanban",
        "waterfall", "itil", "pmp", "project management", "linux", "unix", "rhel",
        "ubuntu", "centos", "windows server", "vmware", "hyper-v"
    ],
    "Business Domains": [
        "banking", "finance", "financial services", "telecom", "telecommunications",
        "healthcare", "pharma", "insurance", "retail", "e-commerce"
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
    """Clean and normalize resume text."""
    if not text:
        return ""
    
    # Normalize unicode characters
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('–', '-')
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '•').replace('\u00a0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
    
    # Fix common PDF extraction issues
    text = re.sub(r'Pres\s*ent', 'Present', text, flags=re.IGNORECASE)
    text = re.sub(r'US\s*A', 'USA', text)
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# ============================================================================
# CONTACT EXTRACTION
# ============================================================================

def extract_contact(text: str) -> Dict[str, str]:
    """Extract contact information from resume."""
    contact = {'email': '', 'phone': '', 'linkedin': '', 'location': ''}
    
    # Email
    match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text)
    if match:
        contact['email'] = match.group(1).lower()
    
    # Phone - multiple formats
    phone_patterns = [
        r'(\+1\s*\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
        r'(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
        r'(\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
        r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, text)
        if match:
            contact['phone'] = match.group(1)
            break
    
    # LinkedIn
    match = re.search(r'linkedin\.com/in/([\w-]+)', text, re.IGNORECASE)
    if match:
        contact['linkedin'] = f"www.linkedin.com/in/{match.group(1)}"
    elif re.search(r'LinkedIn[:\s]+([\w]+)', text):
        match = re.search(r'LinkedIn[:\s]+([\w]+)', text)
        if match and match.group(1).lower() not in ['summary', 'profile', 'in']:
            contact['linkedin'] = f"www.linkedin.com/in/{match.group(1)}"
    
    # Location
    loc_patterns = [
        r'(Philadelphia|Chicago|Dallas|Plano|New York|Bangalore|Bengaluru|Hyderabad)[,\s]+(PA|TX|IL|NY|CA|India|USA|Karnataka)',
        r'\b([A-Z][a-z]+),\s*([A-Z]{2})\b',
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, text)
        if match and not contact['location']:
            contact['location'] = f"{match.group(1)}, {match.group(2)}"
            break
    
    return contact


# ============================================================================
# NAME EXTRACTION
# ============================================================================

def extract_name(text: str) -> Tuple[str, str, str]:
    """Extract first, middle, last name from resume header."""
    lines = [l.strip() for l in text.split('\n') if l.strip()][:10]
    
    for line in lines:
        # Skip headers and contact lines
        if line.lower() in ['resume', 'cv', 'curriculum vitae']:
            continue
        if re.search(r'@|linkedin|phone|email|\d{3}[-.\s]\d{3}', line, re.IGNORECASE):
            continue
        if re.match(r'^(Summary|Skills|Experience|Education)', line, re.IGNORECASE):
            continue
        
        # Clean line
        name = re.sub(r'\s+', ' ', line)
        name = re.sub(r'[\|,].*$', '', name).strip()
        
        parts = name.split()
        if len(parts) >= 2 and all(p[0].isupper() for p in parts if p):
            if len(parts) == 2:
                return parts[0], "", parts[1]
            elif len(parts) == 3:
                return parts[0], parts[1], parts[2]
            else:
                return parts[0], ' '.join(parts[1:-1]), parts[-1]
    
    return "", "", ""


# ============================================================================
# DATE PARSING
# ============================================================================

def parse_date(text: str) -> Tuple[Optional[int], Optional[int], bool]:
    """Parse date string, returns (year, month, is_present)."""
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
# EXPERIENCE EXTRACTION - Multiple Format Support
# ============================================================================

def extract_experiences(text: str) -> List[ExperienceEntry]:
    """Extract work experiences from resume - handles multiple formats."""
    experiences = []
    
    # Full date range pattern
    date_range_pattern = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current)'
    
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for lines containing date ranges
        date_match = re.search(date_range_pattern, line, re.IGNORECASE)
        
        if date_match:
            start_str = date_match.group(1)
            end_str = date_match.group(2)
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if start_year and end_year:
                employer, title, location = "", "", ""
                
                # Check for pipe separator (Jimmy format): "Title | Date" with company on previous line
                # Example: "Senior Technical Manager – Cognizant Infra Services | Jan 2023 – Jun 2025"
                if '|' in line and date_match.start() > 0:
                    pipe_idx = line.rfind('|')
                    title = line[:pipe_idx].strip()
                    title = re.sub(r'\s*[-–]\s*[\w\s]+$', '', title)  # Remove client name if present
                    
                    # Look at previous line for company
                    if i > 0:
                        prev_line = lines[i - 1].strip()
                        # Pattern: "Company – Location"
                        comp_match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+)$', prev_line)
                        if comp_match:
                            employer = comp_match.group(1).strip()
                            location = comp_match.group(2).strip()
                        else:
                            employer = prev_line
                else:
                    # Original format: Company – Title Location Date on same line
                    header_part = line[:date_match.start()].strip()
                    
                    # Format 1: "Company – Title Location" before date
                    comp_match = re.match(
                        r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+?)\s+((?:Philadelphia|Chicago|Dallas|Plano|New York|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]*(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)\s*$',
                        header_part
                    )
                    if comp_match:
                        employer = comp_match.group(1).strip()
                        title = comp_match.group(2).strip()
                        location = comp_match.group(3).strip()
                    else:
                        # Format 2: Just "Company" on this line, title on next
                        employer = header_part.strip()
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if not next_line.startswith(('•', '-', '*')) and len(next_line) < 150:
                                if not re.search(date_range_pattern, next_line, re.IGNORECASE):
                                    loc_match = re.search(
                                        r'((?:Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]*(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)\s*$',
                                        next_line, re.IGNORECASE
                                    )
                                    if loc_match:
                                        title = next_line[:loc_match.start()].strip()
                                        location = loc_match.group(1).strip()
                                    else:
                                        title = next_line
                                    i += 1
                
                # Clean up employer name
                employer = re.sub(r'\s*[-–].*$', '', employer).strip()
                
                # Get responsibilities
                responsibilities = []
                tools = []
                j = i + 1
                while j < len(lines) and j < i + 25:
                    resp_line = lines[j].strip()
                    
                    if re.search(date_range_pattern, resp_line, re.IGNORECASE):
                        break
                    if re.match(r'^(EDUCATION|TECHNICAL|SKILLS|CERTIFICATIONS|KEY\s+ARCH)', resp_line, re.IGNORECASE):
                        break
                    # Skip company/title lines
                    if re.match(r'^[A-Z][A-Za-z\s&]+[-–]', resp_line):
                        break
                    
                    if resp_line.startswith(('•', '-', '*', '–')):
                        resp_text = re.sub(r'^[•\-\*–]\s*', '', resp_line)
                        if len(resp_text) > 30:
                            responsibilities.append(resp_text)
                    
                    j += 1
                
                tools = extract_tools_from_text(' '.join(responsibilities))
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                
                start_date = f"{start_year}-{(start_month or 1):02d}"
                if is_present:
                    now = datetime.now()
                    end_date = f"{now.year}-{now.month:02d}"
                else:
                    end_date = f"{end_year}-{(end_month or 12):02d}"
                
                if employer or title:
                    experiences.append(ExperienceEntry(
                        employer=employer,
                        title=title,
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        duration_months=duration,
                        responsibilities=responsibilities[:10],
                        tools=tools
                    ))
        
        i += 1
    
    # Try "Worked as" format if no experiences found
    if not experiences:
        worked_pattern = r'[Ww]orked\s+as\s+(?:a\s+)?(.+?)\s+in\s+(.+?)\s+from\s+(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|Present)'
        for match in re.finditer(worked_pattern, text):
            title, employer = match.group(1).strip(), match.group(2).strip()
            start_year, start_month, _ = parse_date(match.group(3))
            end_year, end_month, is_present = parse_date(match.group(4))
            
            if start_year and end_year:
                duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
                experiences.append(ExperienceEntry(
                    employer=employer, title=title, location="",
                    start_date=f"{start_year}-{(start_month or 1):02d}",
                    end_date=f"{end_year}-{(end_month or 12):02d}",
                    duration_months=duration, responsibilities=[], tools=[]
                ))
    
    return experiences


def parse_job_header(text: str) -> Tuple[str, str, str]:
    """Parse employer, title, location from job header - handles multiple formats."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    employer, title, location = "", "", ""
    
    # Get last few relevant lines before the date
    relevant_lines = []
    for line in lines[-10:]:
        # Skip bullet points, long lines, and section headers
        if line.startswith(('•', '-', '*')) or len(line) > 200:
            continue
        if re.match(r'^(WORK|EXPERIENCE|EDUCATION|SKILLS|CERTIFICATIONS)', line, re.IGNORECASE):
            continue
        relevant_lines.append(line)
    
    if not relevant_lines:
        return "", "", ""
    
    # Pattern 1: "Company – Title Location" all on one line (Sudheer format)
    # Example: "Comcast – Manager, DevOps, Data & Cloud Philadelphia, PA"
    for line in relevant_lines:
        match = re.match(
            r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*([A-Za-z][\w\s,&]+?)\s+((?:Philadelphia|Chicago|Dallas|Plano|New York|Bangalore|Bengaluru|Hyderabad|India)[,\s]+(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)$',
            line
        )
        if match:
            return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    
    # Pattern 2: Company on first line, Title on second (Khaliq format)
    # Example: "Abbott Laboratories" then "IT Project Manager Chicago, USA"
    company_indicators = ['inc', 'ltd', 'llc', 'corp', 'technologies', 'solutions', 
                         'services', 'systems', 'laboratories', 'consultancy', 'pvt', 'tcs']
    
    for i, line in enumerate(relevant_lines):
        line_lower = line.lower()
        
        # Check if this is a company line
        is_company = (
            any(ci in line_lower for ci in company_indicators) or
            re.match(r'^[A-Z][A-Za-z\s&]+$', line.split('–')[0].split('-')[0].strip())
        )
        
        if is_company and i + 1 < len(relevant_lines):
            employer = line.split('–')[0].split(' - ')[0].strip()
            title_line = relevant_lines[i + 1]
            
            # Extract location from title line
            loc_match = re.search(
                r'((?:Philadelphia|Chicago|Dallas|Plano|New York|Bangalore|Bengaluru|Hyderabad|North Chicago)[,\s]+(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)',
                title_line, re.IGNORECASE
            )
            if loc_match:
                title = title_line[:loc_match.start()].strip()
                location = loc_match.group(1).strip()
            else:
                title = title_line.strip()
            
            return employer, title, location
    
    # Pattern 3: Combined format "Company – Title, Location Date"
    # Example: "Vsion Technologies Inc – NoSQL Database Admin Philadelphia, PA Feb 2016"
    for line in relevant_lines:
        # Remove date part first
        line_clean = re.sub(r'\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}.*$', '', line, flags=re.IGNORECASE)
        
        match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+)$', line_clean)
        if match:
            employer = match.group(1).strip()
            title_loc = match.group(2).strip()
            
            # Try to separate title and location
            loc_match = re.search(
                r'((?:Philadelphia|Chicago|Dallas|Plano|Bangalore|Hyderabad|India)[,\s]+(?:PA|TX|IL|NY|CA|USA|India|Karnataka)?)$',
                title_loc, re.IGNORECASE
            )
            if loc_match:
                title = title_loc[:loc_match.start()].strip()
                location = loc_match.group(1).strip()
            else:
                title = title_loc
            
            if employer and title:
                return employer, title, location
    
    # Fallback: use last non-empty lines
    if len(relevant_lines) >= 2:
        return relevant_lines[-2], relevant_lines[-1], ""
    elif relevant_lines:
        return relevant_lines[-1], "", ""
    
    return "", "", ""


def extract_bullets(text: str) -> List[str]:
    """Extract bullet point responsibilities."""
    bullets = []
    parts = re.split(r'(?:^|\n)\s*[•·▪▸►●○◆◇→\-\*]\s*', text)
    
    for part in parts:
        part = re.sub(r'\s+', ' ', part.strip())
        # Skip short entries and date-like entries
        if len(part) > 40 and not re.match(r'^[A-Z][a-z]+\s+\d{4}', part):
            bullets.append(part)
    
    return bullets


def extract_tools_from_text(text: str) -> List[str]:
    """Extract tools/technologies from text."""
    tools = set()
    text_lower = text.lower()
    
    tool_list = [
        'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'gcp',
        'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
        'mongodb', 'cassandra', 'redis', 'postgresql', 'mysql',
        'kafka', 'spark', 'airflow', 'prometheus', 'grafana', 'elk stack',
        'react', 'angular', 'node.js', 'linux', 'jira', 'servicenow',
        'sonarqube', 'qualys', 'palo alto', 'vmware', 'splunk'
    ]
    
    for tool in tool_list:
        if re.search(rf'\b{re.escape(tool)}\b', text_lower):
            tools.add(tool.title() if len(tool) > 3 else tool.upper())
    
    return sorted(list(tools))


# ============================================================================
# EDUCATION EXTRACTION
# ============================================================================

def extract_education(text: str) -> List[Dict[str, str]]:
    """Extract education entries."""
    education = []
    
    # Find education section
    edu_match = re.search(
        r'EDUCATION[:\s]*\n(.+?)(?:\nTECHNICAL|\nROLES|\nPROJECT|\nKEY|\n_+|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if not edu_match:
        return education
    
    edu_section = edu_match.group(1)
    lines = edu_section.split('\n')
    
    current_entry = {}
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        
        # Skip irrelevant lines
        if re.match(r'^(Roles|Project|•|-|\*)', line, re.IGNORECASE):
            continue
        
        # Pattern: "Degree | Institution | Year"
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                entry = {'degree': parts[0], 'institution': parts[1]}
                year_match = re.search(r'(\d{4})', line)
                if year_match:
                    entry['year'] = year_match.group(1)
                education.append(entry)
                continue
        
        # Pattern: Institution with date range
        date_match = re.search(r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4})', line)
        if date_match:
            inst_match = re.search(r'^(University|[\w\s]+University|JNT\s+University|[\w\s]+Institute)', line, re.IGNORECASE)
            if inst_match:
                current_entry['institution'] = inst_match.group(1).strip()
                year_match = re.search(r'(\d{4})$', line)
                if year_match:
                    current_entry['year'] = year_match.group(1)
            continue
        
        # Degree line (follows institution)
        degree_keywords = ['master', 'bachelor', 'mba', 'mca', 'bca', 'b.tech', 'm.tech', 'b.e', 'm.e', 'ph.d']
        if any(kw in line.lower() for kw in degree_keywords):
            if current_entry.get('institution'):
                current_entry['degree'] = line.strip()
                education.append(current_entry.copy())
                current_entry = {}
            else:
                # Institution and degree on same or nearby lines
                entry = {'degree': line.split('-')[0].split('|')[0].strip()}
                # Look for institution
                for pattern in [r'(University[\w\s,]+)', r'([\w\s]+University)', r'(IGNOU|Pondicherry|Texas A&M)']:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        entry['institution'] = match.group(1).strip()
                        break
                year_match = re.search(r'(\d{4})', line)
                if year_match:
                    entry['year'] = year_match.group(1)
                education.append(entry)
    
    # Add any remaining entry
    if current_entry.get('institution') or current_entry.get('degree'):
        education.append(current_entry)
    
    return education


# ============================================================================
# CERTIFICATION EXTRACTION
# ============================================================================

def extract_certifications(text: str) -> List[str]:
    """Extract certifications."""
    certifications = []
    
    cert_match = re.search(
        r'CERTIFICATIONS?\s*(?:&\s*LEARNING)?[:\s]*\n(.+?)(?:\nWORK|\nEXPERIENCE|\nSKILLS|\nPROJECT|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    
    if cert_match:
        for line in cert_match.group(1).split('\n'):
            line = re.sub(r'^[•·\-\*]\s*', '', line.strip())
            if line and 5 < len(line) < 150:
                if not any(h in line.lower() for h in ['certification', 'learning', 'work experience']):
                    line = re.sub(r'\s*[-–]\s*(?:In Progress|Certification in Progress)\.?$', '', line, flags=re.IGNORECASE)
                    if line:
                        certifications.append(line)
    
    return list(dict.fromkeys(certifications))[:15]


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
    """Calculate skills with experience months."""
    key_skills = {cat: [] for cat in SKILL_CATEGORIES}
    skill_exp: Dict[str, int] = {}
    found_skills: Dict[str, str] = {}
    
    text_lower = text.lower()
    
    # Find all skills in resume
    for cat, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                found_skills[skill] = skill.title() if len(skill) > 3 else skill.upper()
                skill_exp[skill] = 0
    
    # Calculate from experiences
    for exp in experiences:
        job_text = f"{exp.title} {exp.employer} {' '.join(exp.responsibilities)} {' '.join(exp.tools)}".lower()
        
        for skill in found_skills:
            if re.search(rf'\b{re.escape(skill)}\b', job_text):
                skill_exp[skill] = skill_exp.get(skill, 0) + exp.duration_months
    
    # Build output
    for cat, skills in SKILL_CATEGORIES.items():
        cat_skills = []
        for skill in skills:
            if skill in found_skills:
                cat_skills.append({
                    "skill": found_skills[skill],
                    "experience_months": skill_exp.get(skill, 0)
                })
        cat_skills.sort(key=lambda x: x['experience_months'], reverse=True)
        key_skills[cat] = cat_skills
    
    return key_skills


# ============================================================================
# CLAUDE AI VALIDATION - Smart Fallback for Complex Resumes
# ============================================================================

async def validate_and_enhance_with_ai(resume_text: str, parsed_result: Dict) -> Dict:
    """
    Use Claude API to validate and enhance parsed results when critical data is missing.
    This acts as a smart fallback for complex resume formats.
    """
    if not ANTHROPIC_API_KEY:
        return parsed_result
    
    pr = parsed_result.get('parsed_resume', {})
    
    # Check what's missing or suspicious
    issues = []
    if not pr.get('firstname') or not pr.get('lastname'):
        issues.append("name (firstname, lastname)")
    if len(pr.get('experience', [])) < 2:
        issues.append("work experience (need all jobs with employer, title, dates)")
    if not pr.get('title'):
        issues.append("professional title")
    if len(pr.get('education', [])) == 0:
        issues.append("education")
    
    # Only call AI if there are significant issues
    if not issues:
        return parsed_result
    
    try:
        import httpx
        
        prompt = f"""Parse this resume and extract the missing/incomplete information. Return ONLY valid JSON.

MISSING DATA TO EXTRACT: {', '.join(issues)}

RESUME TEXT:
{resume_text[:12000]}

Return JSON with these fields (include ALL work experience entries):
{{
  "firstname": "...",
  "lastname": "...",
  "title": "Professional title from summary",
  "experience": [
    {{
      "Employer": "Company Name",
      "title": "Job Title",
      "location": "City, State/Country",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM",
      "duration_months": N
    }}
  ],
  "education": [
    {{
      "degree": "...",
      "institution": "...",
      "year": "YYYY"
    }}
  ]
}}

Important: Extract ALL jobs listed, not just the first one. Calculate duration_months accurately."""

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
                
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    ai_data = json.loads(json_match.group())
                    
                    # Merge AI results
                    for key in ['firstname', 'lastname', 'title']:
                        if ai_data.get(key) and not pr.get(key):
                            pr[key] = ai_data[key]
                    
                    # Use AI experience if we got more entries
                    ai_exp = ai_data.get('experience', [])
                    if len(ai_exp) > len(pr.get('experience', [])):
                        pr['experience'] = ai_exp
                    
                    # Use AI education if we got entries
                    ai_edu = ai_data.get('education', [])
                    if ai_edu and not pr.get('education'):
                        pr['education'] = ai_edu
                    
                    # Update name field
                    if pr.get('firstname') and pr.get('lastname'):
                        pr['name'] = f"{pr['firstname']} {pr['lastname']}"
                    
                    parsed_result['ai_enhanced'] = True
                    parsed_result['ai_issues_fixed'] = issues
    
    except Exception as e:
        parsed_result['ai_validation_error'] = str(e)
    
    return parsed_result


# ============================================================================
# MAIN PARSING FUNCTION
# ============================================================================

async def parse_resume_full(params: ParseResumeInput) -> str:
    """Parse resume and return structured JSON with optional AI validation."""
    text = normalize_text(params.resume_text)
    
    # Extract basic info
    contact = extract_contact(text)
    firstname, middle, lastname = extract_name(text)
    
    # Extract title from summary
    title = ""
    summary = ""
    summary_match = re.search(
        r'(?:PROFESSIONAL\s+)?SUMMARY[:\s]*\n(.+?)(?:\nSKILLS|\nEXPERIENCE|\nWORK|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if summary_match:
        summary = summary_match.group(1).strip()[:2000]
        # Extract title
        for pattern in [
            r'^([\w\s]+(?:Engineer|Manager|Architect|Developer|Lead|Director|Analyst|Consultant))',
            r'^(IT\s+Project\s+Manager)',
            r'^(Scrum\s+Master)',
            r'(DevOps[\w\s,&]+Engineer)',
        ]:
            match = re.search(pattern, summary, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                break
    
    # Extract experiences
    experiences = extract_experiences(text)
    
    # Build result
    name = f"{firstname} {middle} {lastname}".replace('  ', ' ').strip() if middle else f"{firstname} {lastname}"
    
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
            "technical_skills": extract_technical_skills(text),
            "key_skills": calculate_key_skills(text, experiences),
            "education": extract_education(text),
            "certifications": extract_certifications(text),
            "experience": [
                {
                    "Employer": e.employer,
                    "title": e.title,
                    "location": e.location,
                    "start_date": e.start_date,
                    "end_date": e.end_date,
                    "duration_months": e.duration_months,
                    "responsibilities": e.responsibilities,
                    "tools": e.tools
                }
                for e in experiences
            ],
            "filename": params.filename or ""
        }
    }
    
    # AI validation for complex cases
    if params.use_ai_validation and ANTHROPIC_API_KEY:
        result = await validate_and_enhance_with_ai(text, result)
    
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
