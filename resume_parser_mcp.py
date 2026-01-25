"""
Resume Parser MCP Server - Production Grade
============================================
Enterprise-level resume parsing with exact output format matching.
"""

import json
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass
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
# CONSTANTS
# ============================================================================

NAME_PREFIXES = {'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'professor', 'sir', 'madam'}
NAME_SUFFIXES = {'jr', 'jr.', 'sr', 'sr.', 'i', 'ii', 'iii', 'iv', 'v', 
                 'phd', 'ph.d', 'ph.d.', 'md', 'm.d', 'm.d.', 'esq', 'esq.',
                 'mba', 'cpa', 'pe', 'pmp', 'csm', 'cissp'}

MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

SECTION_HEADERS = {
    'experience': ['professional experience', 'work experience', 'experience', 'employment', 
                   'career history', 'employment history', 'professional background', 'work history'],
    'education': ['education', 'academic background', 'academic qualifications', 'qualifications',
                  'educational background', 'academic history', 'degrees'],
    'skills': ['skills', 'technical skills', 'core competencies', 'competencies', 'expertise',
               'technologies', 'tools', 'proficiencies', 'areas of expertise', 'key skills'],
    'summary': ['professional summary', 'summary', 'executive summary', 'profile', 
                'about', 'objective', 'career objective', 'professional profile'],
    'certifications': ['certifications', 'certificates', 'licenses', 'professional certifications',
                       'credentials', 'accreditations'],
}

# Skill categories matching expected output format
SKILL_CATEGORIES = {
    "Data Engineering": [
        "data warehousing", "dwh", "etl", "elt", "data migration",
        "informatica", "ssis", "talend", "data integration", "data pipelines",
        "apache airflow", "airflow", "dagster", "prefect",
        "apache spark", "spark", "pyspark", "apache kafka", "kafka",
        "data lake", "snowflake", "databricks", "data modeling",
        "star schema", "snowflake schema", "dbt", "glue", "redshift", "bigquery"
    ],
    "Programming": [
        "python", "java", "javascript", "typescript", "c++", "c#",
        "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala",
        "r", "perl", "bash", "shell", "powershell", "sql",
        "html", "css", "groovy", "scripting", "automation scripting",
        "bdd framework", "tdd", "gherkin", "gherkins", "plsql", "pl/sql"
    ],
    "Databases": [
        "database management", "database testing", "database",
        "postgresql", "postgres", "mysql", "mongodb", "redis",
        "elasticsearch", "cassandra", "dynamodb", "oracle", "oracle server",
        "sql server", "mssql", "sqlite", "neo4j", "sql developer", "nosql", "db2"
    ],
    "Cloud": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp",
        "google cloud", "ibm cloud", "oracle cloud", "heroku",
        "ec2", "s3", "lambda", "rds", "ecs", "eks",
        "azure functions", "cosmos db", "blob storage", "aks",
        "cloud functions", "cloud run", "gke",
        "vmware", "vmware vsphere", "virtualization", "cloud security",
        "cloud computing"
    ],
    "DevOps": [
        "docker", "kubernetes", "k8s", "helm", "terraform",
        "ansible", "puppet", "chef", "jenkins", "gitlab ci", "github actions",
        "circleci", "ci/cd", "cicd", "continuous integration", "continuous delivery",
        "prometheus", "grafana", "datadog", "splunk",
        "git", "github", "gitlab", "bitbucket",
        "azure devops", "devops", "agile", "waterfall", "scrum", "kanban"
    ],
    "Data Science & Visualization": [
        "machine learning", "ml", "deep learning", "artificial intelligence", "ai",
        "tensorflow", "pytorch", "keras", "scikit-learn",
        "nlp", "natural language processing", "computer vision",
        "data science", "data analysis", "statistics",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "tableau", "power bi", "powerbi", "looker", "qlik",
        "data visualization", "bi reporting", "dashboards", "reporting",
        "executive dashboards", "sla reporting", "analytics"
    ],
    "Other Tools": [
        "selenium", "pytest", "unittest", "jest", "cypress", "postman",
        "api testing", "automation testing", "manual testing", "functional testing",
        "regression testing", "ui testing", "qa", "quality assurance",
        "software testing", "test automation", "hp alm", "hpalm", "zephyr",
        "cucumber", "behave", "cucumber/behave", "pycharm", "vscode",
        "servicenow", "power automate", "ms project", "clarity-ppm",
        "risk management", "pmo governance", "itil", "project management",
        "program management", "portfolio management", "putty", "unix", "linux",
        "rest api", "soap", "microservices", "api", "web services",
        "jira", "confluence", "unix/linux"
    ],
    "Business Domains": [
        "telecom", "telecommunications", "bss", "oss", "bss/oss", "telecom bss/oss",
        "banking", "finance", "financial services", "fintech",
        "insurance", "healthcare", "pharma", "pharmaceutical", "pharmaceuticals",
        "retail", "e-commerce", "ecommerce", "crm", "erp",
        "education", "publishing", "media", "entertainment",
        "aviation", "logistics", "supply chain", "manufacturing",
        "payment", "payments", "billing"
    ]
}

# Flatten for quick lookup
ALL_SKILLS: Set[str] = set()
for skills in SKILL_CATEGORIES.values():
    ALL_SKILLS.update(skills)

# Common tools for extraction per job
COMMON_TOOLS = [
    "python", "java", "sql", "selenium", "pytest", "jenkins", "git", "github",
    "docker", "kubernetes", "aws", "azure", "gcp", "jira", "confluence",
    "postman", "tableau", "power bi", "excel", "terraform", "ansible",
    "mongodb", "postgresql", "mysql", "oracle", "redis", "kafka",
    "spark", "airflow", "snowflake", "databricks", "tensorflow", "pytorch",
    "pandas", "numpy", "react", "angular", "vue", "node", "nodejs",
    "spring", "spring boot", "django", "flask", "fastapi", ".net", "c#",
    "informatica", "ssis", "talend", "unix", "linux", "windows",
    "servicenow", "power automate", "azure devops", "gitlab", "bitbucket",
    "vmware", "vmware vsphere", "agile", "scrum", "waterfall", "kanban",
    "hp alm", "zephyr", "cucumber", "behave", "gherkin", "pycharm",
    "ibm cloud", "unix/linux", "scripting", "automation scripting", "itil"
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ParsedName:
    full_name: str
    first_name: str
    middle_name: Optional[str] = None
    last_name: str = ""
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ParsedDate:
    original: str
    year: int
    month: Optional[int] = None
    is_present: bool = False
    
    def to_date_string(self) -> str:
        if self.is_present:
            now = datetime.now()
            return f"{now.year}-{now.month:02d}"
        if self.month:
            return f"{self.year}-{self.month:02d}"
        return f"{self.year}-01"


@dataclass 
class DateRange:
    start: ParsedDate
    end: ParsedDate
    duration_months: int
    is_current: bool


@dataclass
class ExperienceEntry:
    employer: str
    title: str
    location: str
    start_date: str
    end_date: str
    duration_months: int
    responsibilities: List[str]
    tools: List[str]


# ============================================================================
# PYDANTIC INPUT MODELS
# ============================================================================

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class ParseResumeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    
    resume_text: str = Field(..., min_length=50, max_length=100000)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON)
    filename: Optional[str] = Field(default=None)

    @field_validator('resume_text')
    @classmethod
    def clean_text(cls, v: str) -> str:
        return normalize_text(v)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    
    # Keep special characters that might be meaningful
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2022', '•')
    text = text.replace('\u00a0', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\t', ' ')
    
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# ============================================================================
# NAME PARSING
# ============================================================================

def parse_name(text: str) -> ParsedName:
    if not text:
        return ParsedName(full_name="", first_name="", last_name="", confidence=0.0)
    
    name = text.strip()
    
    # Skip headers
    if name.lower() in ['resume', 'cv', 'curriculum vitae', 'profile', 'biodata']:
        return ParsedName(full_name="", first_name="", last_name="", confidence=0.0)
    
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[\|,].*$', '', name)
    name = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', name)
    name = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', name)
    name = re.sub(r'linkedin\.com\S*', '', name, flags=re.IGNORECASE)
    name = name.strip()
    
    prefix = None
    for p in NAME_PREFIXES:
        match = re.match(rf'^({p}\.?)\s+', name, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            name = name[match.end():].strip()
            break
    
    suffix = None
    for s in NAME_SUFFIXES:
        match = re.search(rf',?\s*({s}\.?)$', name, re.IGNORECASE)
        if match:
            suffix = match.group(1)
            name = name[:match.start()].strip().rstrip(',')
            break
    
    parts = name.split()
    
    if len(parts) == 0:
        return ParsedName(full_name=text, first_name="", last_name="", confidence=0.1)
    if len(parts) == 1:
        return ParsedName(full_name=text, first_name=parts[0], last_name="",
                         prefix=prefix, suffix=suffix, confidence=0.5)
    if len(parts) == 2:
        return ParsedName(full_name=text, first_name=parts[0], last_name=parts[1],
                         prefix=prefix, suffix=suffix, confidence=0.9)
    
    return ParsedName(full_name=text, first_name=parts[0], 
                     middle_name=' '.join(parts[1:-1]) if len(parts) > 2 else None,
                     last_name=parts[-1], prefix=prefix, suffix=suffix, confidence=0.85)


# ============================================================================
# DATE PARSING
# ============================================================================

def parse_date_string(text: str) -> Optional[ParsedDate]:
    if not text:
        return None
    
    text = text.strip().lower()
    original = text
    
    if any(p in text for p in ['present', 'current', 'now', 'ongoing', 'today', 'till date']):
        now = datetime.now()
        return ParsedDate(original=original, year=now.year, month=now.month, is_present=True)
    
    # Month Year pattern
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*[,.]?\s*(\d{4})', text)
    if match:
        month = MONTH_MAP.get(match.group(1)[:3].lower())
        year = int(match.group(2))
        if month and 1900 <= year <= 2100:
            return ParsedDate(original=original, year=year, month=month)
    
    # MM/YYYY or MM-YYYY
    match = re.search(r'(\d{1,2})[/\-](\d{4})', text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return ParsedDate(original=original, year=year, month=month)
    
    # YYYY-MM
    match = re.search(r'(\d{4})[/\-](\d{1,2})', text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return ParsedDate(original=original, year=year, month=month)
    
    # Just year
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if match:
        return ParsedDate(original=original, year=int(match.group(1)))
    
    return None


def parse_date_range(text: str) -> Optional[DateRange]:
    if not text:
        return None
    
    text = normalize_text(text)
    
    parts = None
    for sep in [' – ', ' - ', ' to ', ' until ', '–', '-']:
        if sep in text.lower():
            parts = re.split(re.escape(sep), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                break
    
    if not parts or len(parts) != 2:
        matches = re.findall(r'((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{4}|\d{1,2}[/\-]\d{4}|\d{4})', text, re.IGNORECASE)
        if len(matches) >= 2:
            parts = [matches[0], matches[-1]]
        else:
            return None
    
    start = parse_date_string(parts[0].strip())
    end = parse_date_string(parts[1].strip())
    
    if not start or not end:
        return None
    
    start_dt = datetime(start.year, start.month or 1, 1)
    end_dt = datetime(end.year, end.month or 12, 1)
    
    if end_dt < start_dt:
        start, end = end, start
        start_dt, end_dt = end_dt, start_dt
    
    delta = relativedelta(end_dt, start_dt)
    duration = max(1, delta.years * 12 + delta.months)
    
    return DateRange(start=start, end=end, duration_months=duration, is_current=end.is_present)


# ============================================================================
# SECTION DETECTION
# ============================================================================

def detect_sections(text: str) -> Dict[str, str]:
    lines = text.split('\n')
    sections = {}
    current_section = 'header'
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        line_clean = re.sub(r'[^\w\s]', '', line_lower)
        
        found_section = None
        for section_type, headers in SECTION_HEADERS.items():
            for header in headers:
                if line_clean == header or line_lower.startswith(header + ':'):
                    found_section = section_type
                    break
                if len(line_clean) < 50 and header in line_clean and len(line_clean.split()) <= 4:
                    found_section = section_type
                    break
            if found_section:
                break
        
        if found_section:
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            current_section = found_section
            current_content = []
        else:
            current_content.append(line)
    
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections


# ============================================================================
# SKILL EXTRACTION
# ============================================================================

def format_skill_name(skill: str) -> str:
    """Format skill name for display."""
    # Special cases
    special_cases = {
        'sql': 'SQL', 'html': 'HTML', 'css': 'CSS', 'api': 'API',
        'aws': 'AWS', 'gcp': 'GCP', 'ci/cd': 'CI/CD', 'cicd': 'CI/CD',
        'etl': 'ETL', 'dwh': 'DWH', 'plsql': 'PLSQL', 'pl/sql': 'PL/SQL',
        'hp alm': 'HP ALM', 'hpalm': 'HP ALM', 'ui': 'UI', 'qa': 'QA',
        'bss': 'BSS', 'oss': 'OSS', 'bss/oss': 'BSS/OSS',
        'telecom bss/oss': 'Telecom BSS/OSS', 'crm': 'CRM', 'erp': 'ERP',
        'itil': 'ITIL', 'pmo governance': 'PMO Governance',
        'vmware vsphere': 'VMware vSphere', 'ibm cloud': 'IBM Cloud',
        'azure devops': 'Azure DevOps', 'power bi': 'Power BI',
        'unix/linux': 'Unix/Linux', 'cucumber/behave': 'Cucumber/Behave',
        'bdd framework': 'BDD Framework', 'sla reporting': 'SLA Reporting',
        'executive dashboards': 'Executive Dashboards',
        'bi reporting': 'BI Reporting', 'database management': 'Database Management',
        'database testing': 'Database Testing', 'risk management': 'Risk Management',
        'data warehousing': 'Data Warehousing', 'data migration': 'Data Migration',
    }
    
    skill_lower = skill.lower()
    if skill_lower in special_cases:
        return special_cases[skill_lower]
    
    # General title case
    return skill.title()


def extract_technical_skills(text: str) -> List[str]:
    """Extract flat list of technical skills as they appear in the resume."""
    skills_found = set()
    text_lower = text.lower()
    
    # Find skills section
    skills_section = ""
    for line in text.split('\n'):
        line_lower = line.strip().lower()
        if any(h in line_lower for h in ['technical skills', 'skills', 'tools', 'technologies']):
            # Get content after this header
            idx = text.lower().find(line_lower)
            if idx >= 0:
                remaining = text[idx:]
                # Find next major section
                next_section = re.search(r'\n(?:EDUCATION|EXPERIENCE|CERTIFICATIONS|PROFESSIONAL)', remaining[len(line):], re.IGNORECASE)
                if next_section:
                    skills_section = remaining[len(line):next_section.start() + len(line)]
                else:
                    skills_section = remaining[len(line):len(line) + 1500]
                break
    
    # Extract skills from skills section first
    if skills_section:
        # Split by common delimiters
        items = re.split(r'[,|•·\n]', skills_section)
        for item in items:
            item = item.strip()
            # Clean up item
            item = re.sub(r'^[-–•*]\s*', '', item)
            item = re.sub(r'^\d+\.\s*', '', item)
            if item and len(item) > 1 and len(item) < 50:
                # Check if it's a known skill or looks like one
                item_lower = item.lower()
                for skill in ALL_SKILLS:
                    if skill in item_lower or item_lower in skill:
                        skills_found.add(format_skill_name(item))
                        break
                # Also check for common tool patterns
                for tool in COMMON_TOOLS:
                    if tool in item_lower:
                        skills_found.add(format_skill_name(tool))
    
    # Also scan full text for skills
    for skill in ALL_SKILLS:
        if re.search(rf'\b{re.escape(skill)}\b', text_lower):
            skills_found.add(format_skill_name(skill))
    
    return sorted(list(skills_found))


def extract_tools_from_text(text: str) -> List[str]:
    """Extract tools mentioned in specific text block."""
    text_lower = text.lower()
    tools = set()
    
    for tool in COMMON_TOOLS:
        if re.search(rf'\b{re.escape(tool)}\b', text_lower):
            tools.add(format_skill_name(tool))
    
    return sorted(list(tools))


# ============================================================================
# EXPERIENCE EXTRACTION
# ============================================================================

def extract_experiences(text: str) -> List[ExperienceEntry]:
    """Extract work experiences supporting multiple resume formats."""
    experiences = []
    
    # Format 1: "Worked as a [title] in [company] from [date] to [date]"
    worked_pattern = r'[Ww]orked\s+as\s+(?:a\s+)?(.+?)\s+in\s+(.+?)\s+from\s+(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|[Pp]resent|[Cc]urrent)'
    worked_matches = list(re.finditer(worked_pattern, text))
    
    # Format 2: Standard format - Company + Location, then Title | Date
    # Pattern: "Company Name – Location" or "Company Name - Location"
    # Then: "Title | Date Range" or "Title – Date Range"
    standard_pattern = r'^([A-Z][A-Za-z\s&.,]+(?:Ltd|Inc|Corp|LLC|Pvt|Technologies|Solutions|Systems|Services|India)?[A-Za-z\s.,]*)\s*[–-]\s*([A-Za-z\s,]+)$'
    
    # Find PROFESSIONAL EXPERIENCE or WORK EXPERIENCE section
    exp_section = ""
    exp_match = re.search(r'(?:PROFESSIONAL\s+EXPERIENCE|WORK\s+EXPERIENCE|EXPERIENCE)[:\s]*\n(.+?)(?:\nEDUCATION|\nSKILLS|\nCERTIFICATION|\Z)', text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        exp_section = exp_match.group(1)
    
    # Try Format 1 first (Worked as...)
    if worked_matches:
        experiences = extract_experiences_worked_format(text, worked_matches)
    
    # Try Format 2 (Standard format)
    if not experiences and exp_section:
        experiences = extract_experiences_standard_format(exp_section, text)
    
    return experiences


def extract_experiences_worked_format(text: str, worked_matches) -> List[ExperienceEntry]:
    """Extract experiences from 'Worked as X in Y from Z to W' format."""
    experiences = []
    
    # Find detailed WORK EXPERIENCE section
    work_exp_match = re.search(r'WORK\s+EXPERIENCE[:\s]*\n(.+?)(?:\nEDUCATION|\nSKILLS|\nCERTIFICATION|\Z)', text, re.IGNORECASE | re.DOTALL)
    work_exp_text = work_exp_match.group(1) if work_exp_match else ""
    
    # Build company blocks from detailed section
    company_blocks = parse_company_blocks(work_exp_text) if work_exp_text else {}
    
    for match in worked_matches:
        title = match.group(1).strip()
        employer = match.group(2).strip()
        start_str = match.group(3).strip()
        end_str = match.group(4).strip()
        
        start = parse_date_string(start_str)
        end = parse_date_string(end_str)
        
        if start and end:
            start_dt = datetime(start.year, start.month or 1, 1)
            end_dt = datetime(end.year, end.month or 12, 1)
            delta = relativedelta(end_dt, start_dt)
            duration = max(1, delta.years * 12 + delta.months)
            
            responsibilities = []
            tools = []
            
            # Match with detailed company block
            for company_name, block_content in company_blocks.items():
                if (employer.lower() in company_name.lower() or 
                    company_name.lower() in employer.lower() or
                    any(word in company_name.lower() for word in employer.lower().split())):
                    responsibilities = extract_responsibilities(block_content)
                    tools = extract_tools_from_text(block_content)
                    break
            
            experiences.append(ExperienceEntry(
                employer=employer,
                title=title,
                location="",
                start_date=start.to_date_string(),
                end_date=end.to_date_string(),
                duration_months=duration,
                responsibilities=responsibilities,
                tools=tools
            ))
    
    return experiences


def extract_experiences_standard_format(exp_section: str, full_text: str) -> List[ExperienceEntry]:
    """Extract experiences from standard format (Company - Location, Title | Date)."""
    experiences = []
    lines = exp_section.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for company line (Company Name – Location)
        company_match = re.match(r'^([A-Z][A-Za-z\s&.,]+(?:Ltd|Inc|Corp|LLC|Pvt)?[A-Za-z\s.,]*)\s*[–-]\s*([A-Za-z\s,]+)$', line)
        
        if company_match:
            employer = company_match.group(1).strip()
            location = company_match.group(2).strip()
            
            # Next line should be title | date
            if i + 1 < len(lines):
                title_line = lines[i + 1].strip()
                # Match: Title | Date or Title – Date
                title_match = re.match(r'^(.+?)\s*[|–-]\s*(\w+\s+\d{4}\s*[–-]\s*(?:\w+\s+\d{4}|Present|Current))$', title_line, re.IGNORECASE)
                
                if title_match:
                    title = title_match.group(1).strip()
                    date_range_str = title_match.group(2).strip()
                    date_range = parse_date_range(date_range_str)
                    
                    if date_range:
                        # Collect responsibilities (bullet points following)
                        responsibilities = []
                        j = i + 2
                        while j < len(lines):
                            resp_line = lines[j].strip()
                            # Check if this is a bullet point
                            if resp_line.startswith(('•', '-', '*', '–')):
                                resp_text = re.sub(r'^[•\-*–]\s*', '', resp_line)
                                if resp_text:
                                    responsibilities.append(resp_text)
                            # Check if we've hit the next company or section
                            elif re.match(r'^[A-Z][A-Za-z\s&.,]+[–-]', resp_line) or \
                                 re.match(r'^(?:EDUCATION|SKILLS|CERTIFICATION)', resp_line, re.IGNORECASE):
                                break
                            elif resp_line and not resp_line.startswith(('•', '-', '*')):
                                # Could be continuation or new section
                                if len(resp_line) > 50 and not re.search(r'\d{4}', resp_line):
                                    responsibilities.append(resp_line)
                            j += 1
                        
                        tools = extract_tools_from_text(' '.join(responsibilities))
                        
                        experiences.append(ExperienceEntry(
                            employer=employer,
                            title=title,
                            location=location,
                            start_date=date_range.start.to_date_string(),
                            end_date=date_range.end.to_date_string(),
                            duration_months=date_range.duration_months,
                            responsibilities=responsibilities,
                            tools=tools
                        ))
                        
                        i = j
                        continue
        i += 1
    
    return experiences


def parse_company_blocks(text: str) -> Dict[str, str]:
    """Parse company blocks from detailed work experience section."""
    blocks = {}
    lines = text.split('\n')
    
    current_company = None
    current_content = []
    
    # Company indicators
    company_patterns = [
        r'Ltd\.?', r'Inc\.?', r'Corp\.?', r'LLC', r'Pvt\.?', r'Limited',
        r'Technologies', r'Solutions', r'Services', r'Systems', r'India',
        r'Mahindra', r'Capgemini', r'Cognizant', r'IBM', r'TCS', r'Infosys',
        r'Wipro', r'Accenture', r'Dell', r'EMC', r'IMG'
    ]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Check if this line is a company name
        is_company = (
            len(line_stripped) < 80 and
            not line_stripped.startswith(('•', '-', '*', '–')) and
            any(re.search(p, line_stripped, re.IGNORECASE) for p in company_patterns) and
            'Key Responsibilities' not in line_stripped and
            'Project' not in line_stripped
        )
        
        if is_company:
            if current_company and current_content:
                blocks[current_company] = '\n'.join(current_content)
            current_company = line_stripped
            current_content = []
        else:
            current_content.append(line_stripped)
    
    if current_company and current_content:
        blocks[current_company] = '\n'.join(current_content)
    
    return blocks


def extract_responsibilities(text: str) -> List[str]:
    """Extract bullet points and responsibilities."""
    responsibilities = []
    
    # Look for Key Responsibilities section
    key_resp_match = re.search(r'Key\s+Responsibilities[:\s]*\n(.+?)(?:\n\n|\nProject|\nClient|\Z)', text, re.IGNORECASE | re.DOTALL)
    if key_resp_match:
        text = key_resp_match.group(1)
    
    # Split by bullet points
    parts = re.split(r'(?:^|\n)\s*(?:[•·▪▸►●○◆◇→\-\*]|\d+[.\)])\s*', text)
    
    for part in parts:
        part = re.sub(r'\s+', ' ', part.strip())
        if len(part) > 20 and not re.match(r'^\d{4}|^[A-Z][a-z]+\s+\d{4}', part):
            responsibilities.append(part.rstrip('.') + '.')
    
    return responsibilities[:10]


# ============================================================================
# EDUCATION EXTRACTION
# ============================================================================

def extract_education(text: str) -> List[Dict[str, str]]:
    """Extract education entries."""
    education = []
    
    edu_match = re.search(r'EDUCATION[:\s]*\n(.+?)(?:\nCERTIFICATION|\nEXPERIENCE|\nSKILLS|\Z)', text, re.IGNORECASE | re.DOTALL)
    edu_section = edu_match.group(1) if edu_match else ""
    
    if not edu_section:
        return education
    
    lines = edu_section.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        entry = {}
        
        # Look for degree
        degree_patterns = [
            (r"(Master\s+of\s+Computer\s+Applications?\s*\(MCA\))", "Master of Computer Applications (MCA)"),
            (r"(MCA)", "MCA"),
            (r"(ME\s+Data\s+Science)", "ME Data Science"),
            (r"(M\.?E\.?|M\.?S\.?|M\.?A\.?|M\.?B\.?A\.?|Master)", None),
            (r"(B\.?E\.?|B\.?S\.?|B\.?A\.?|B\.?Tech|Bachelor)", None),
            (r"(Ph\.?D\.?|Doctorate)", None),
        ]
        
        for pattern, replacement in degree_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                # Get the full degree description
                entry['degree'] = line.split('–')[0].split('-')[0].split('|')[0].strip()
                break
        
        # Look for institution
        inst_patterns = [
            r'(IGNOU|Pune University|VIT|IIT|NIT|BITS)',
            r'(University[\w\s,]+)',
            r'(Institute[\w\s,]+)',
            r'(College[\w\s,]+)',
        ]
        for pattern in inst_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match and 'institution' not in entry:
                entry['institution'] = match.group(1).strip()
        
        # Look for year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', line)
        if year_match:
            entry['year'] = year_match.group(1)
        
        if entry.get('degree'):
            education.append(entry)
    
    return education


# ============================================================================
# CERTIFICATION EXTRACTION
# ============================================================================

def extract_certifications(text: str) -> List[str]:
    """Extract certifications."""
    certifications = []
    
    cert_match = re.search(r'CERTIFICATIONS?[:\s]*\n(.+?)(?:\nEDUCATION|\nEXPERIENCE|\nSKILLS|\Z)', text, re.IGNORECASE | re.DOTALL)
    cert_section = cert_match.group(1) if cert_match else ""
    
    if cert_section:
        # Split by newlines and common delimiters
        items = re.split(r'[\n|]', cert_section)
        for item in items:
            item = item.strip()
            item = re.sub(r'^[•·\-\*]\s*', '', item)
            if item and 3 < len(item) < 100:
                # Skip section headers
                if not any(h in item.lower() for h in ['certification', 'certificate', 'credential']):
                    # Clean up "In Progress" markers
                    item = re.sub(r'\s*[–-]\s*In Progress$', '', item, flags=re.IGNORECASE)
                    if item:
                        certifications.append(item)
    
    return list(dict.fromkeys(certifications))[:20]  # Remove duplicates, keep order


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
    for pattern in [r'(\+\d{1,3}\s*\d{3}\s*\d{3}\s*\d{4})', r'(\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
                   r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})', r'(\d{10})']:
        match = re.search(pattern, text)
        if match:
            contact['phone'] = match.group(1)
            break
    
    # LinkedIn
    match = re.search(r'((?:www\.)?linkedin\.com/in/[\w-]+)', text, re.IGNORECASE)
    if match:
        contact['linkedin'] = match.group(1)
    
    # Location - usually in header line
    header_lines = text.split('\n')[:5]
    for line in header_lines:
        # Pattern: City, State or City, Country
        loc_match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', line)
        if loc_match and not contact['location']:
            contact['location'] = f"{loc_match.group(1)}, {loc_match.group(2)}"
    
    return contact


# ============================================================================
# KEY SKILLS WITH EXPERIENCE CALCULATION
# ============================================================================

def calculate_key_skills(text: str, experiences: List[ExperienceEntry]) -> Dict[str, List[Dict]]:
    """Calculate skills with experience months by category."""
    key_skills = {cat: [] for cat in SKILL_CATEGORIES}
    skill_exp: Dict[str, int] = {}
    
    text_lower = text.lower()
    found_skills: Dict[str, str] = {}  # normalized -> display name
    
    # Find all skills in resume
    for cat, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if re.search(rf'\b{re.escape(skill)}\b', text_lower):
                found_skills[skill] = format_skill_name(skill)
                skill_exp[skill] = 0
    
    # Calculate experience from jobs
    for exp in experiences:
        job_parts = [
            exp.title.lower(), 
            exp.employer.lower(),
            ' '.join(exp.responsibilities).lower(),
            ' '.join(exp.tools).lower()
        ]
        job_text = ' '.join(job_parts)
        
        for skill in found_skills:
            # Direct match
            if re.search(rf'\b{re.escape(skill)}\b', job_text):
                skill_exp[skill] = skill_exp.get(skill, 0) + exp.duration_months
            # Match via tools
            elif any(skill in t.lower() for t in exp.tools):
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
        
        # Sort by experience descending
        cat_skills.sort(key=lambda x: x['experience_months'], reverse=True)
        key_skills[cat] = cat_skills
    
    return key_skills


# ============================================================================
# MAIN PARSING FUNCTION
# ============================================================================

async def parse_resume_full(params: ParseResumeInput) -> str:
    """Parse resume and return structured JSON."""
    text = normalize_text(params.resume_text)
    sections = detect_sections(text)
    
    header = sections.get('header', '')
    header_lines = [l.strip() for l in header.split('\n') if l.strip()]
    
    # Find name (skip "Resume" header)
    parsed_name = ParsedName(full_name="", first_name="", last_name="", confidence=0.0)
    for line in header_lines:
        if line.lower() in ['resume', 'cv', 'curriculum vitae']:
            continue
        if not re.search(r'@|linkedin|github|\d{3}[-.\s]\d{3}|mob:|email:|phone:|plano|bangalore|texas|karnataka', line, re.IGNORECASE):
            parsed_name = parse_name(line)
            if parsed_name.first_name:
                break
    
    # Find title
    title = ""
    # First check summary section for title
    summary_text = sections.get('summary', '')
    if summary_text:
        # Look for title pattern at start
        title_match = re.match(r'^((?:Project|Senior|Lead|Principal|Staff)\s*(?:Manager|Engineer|Developer|Analyst|Consultant|Architect|Specialist|Director))', summary_text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1)
    
    # Also check header lines
    if not title:
        for line in header_lines:
            if line.lower() in ['resume', 'cv', parsed_name.full_name.lower()]:
                continue
            if '@' in line or 'linkedin' in line.lower() or re.search(r'\d{3}[-.\s]\d{3}', line):
                continue
            title_keywords = ['manager', 'engineer', 'developer', 'analyst', 'consultant', 
                            'architect', 'lead', 'director', 'specialist', 'scientist']
            if any(kw in line.lower() for kw in title_keywords):
                title = re.split(r'[|–-]', line)[0].strip()
                break
    
    # Extract contact
    contact = extract_contact(text)
    
    # Extract experiences
    experiences = extract_experiences(text)
    
    # Extract summary
    summary = sections.get('summary', '').strip()
    if not summary:
        # Try to get from header or qualification section
        qual_match = re.search(r'(?:TECHNICAL\s+)?QUALIFICATION\s+HEADLINE[:\s]*\n(.+?)(?:\n\n|\nEXPERIENCE|\nSKILLS)', text, re.IGNORECASE | re.DOTALL)
        if qual_match:
            summary = qual_match.group(1).strip()
    
    # Build result
    result = {
        "parsed_resume": {
            "firstname": parsed_name.first_name,
            "lastname": parsed_name.last_name,
            "name": f"{parsed_name.first_name} {parsed_name.last_name}".strip(),
            "title": title,
            "location": contact.get('location', ''),
            "phone_number": contact.get('phone', ''),
            "email": contact.get('email', ''),
            "linkedin": contact.get('linkedin', ''),
            "summary": summary[:3000] if summary else "",
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
