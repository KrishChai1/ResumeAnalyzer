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
    file_path: Optional[str] = Field(default=None)  # Path to DOCX file for table extraction
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
    """Extract text from DOCX including tables and text boxes - handles all formats."""
    from docx import Document
    import re as regex
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
    
    # Extract text from text boxes (important for sidebar layouts like Nageswara)
    try:
        xml_str = doc.element.xml
        # Find all text box content
        pattern = r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>'
        matches = regex.findall(pattern, xml_str, regex.DOTALL)
        
        for match in matches:
            # Extract actual text from the XML
            text_pattern = r'<w:t[^>]*>([^<]+)</w:t>'
            texts = regex.findall(text_pattern, match)
            if texts:
                combined = ' '.join(texts)
                # Clean up excessive whitespace
                combined = ' '.join(combined.split())
                if combined and len(combined) > 5:
                    all_text.append(combined)
    except Exception:
        pass  # Fall back to standard extraction
    
    return '\n'.join(all_text)


def extract_from_docx_textboxes(file_path: str) -> Dict:
    """
    Extract structured data from DOCX text boxes (sidebar layouts).
    Returns dict with: contact, education, certifications, skills
    """
    import re as regex
    from docx import Document
    
    result = {
        'phone': None,
        'email': None,
        'linkedin': None,
        'education': [],
        'certifications': [],
        'skills': []
    }
    
    seen_education = set()  # For deduplication
    
    try:
        doc = Document(file_path)
        xml_str = doc.element.xml
        
        # Find all text box content
        pattern = r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>'
        matches = regex.findall(pattern, xml_str, regex.DOTALL)
        
        seen_texts = set()
        
        for match in matches:
            # Extract actual text from the XML
            text_pattern = r'<w:t[^>]*>([^<]+)</w:t>'
            texts = regex.findall(text_pattern, match)
            if texts:
                combined = ' '.join(texts)
                combined = ' '.join(combined.split())
                
                # Skip duplicates
                if combined in seen_texts:
                    continue
                seen_texts.add(combined)
                
                # Extract phone
                phone_match = regex.search(r'Mobile[:\s]*([+\d\s\-\(\)]{10,20})', combined, regex.IGNORECASE)
                if phone_match and not result['phone']:
                    result['phone'] = phone_match.group(1).strip()
                
                # Extract email
                email_match = regex.search(r'Mail[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', combined, regex.IGNORECASE)
                if email_match and not result['email']:
                    result['email'] = email_match.group(1).strip()
                
                # Also check for plain email format
                if not result['email']:
                    email_match = regex.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', combined)
                    if email_match:
                        result['email'] = email_match.group(1).strip()
                
                # Extract LinkedIn
                linkedin_match = regex.search(r'LinkedIn[:\s]*([a-zA-Z0-9\-]+)', combined, regex.IGNORECASE)
                if linkedin_match and not result['linkedin']:
                    result['linkedin'] = linkedin_match.group(1).strip()
                
                # Extract education (Qualification/Masters/Bachelor) - with deduplication
                edu_match = regex.search(r'(?:Qualification|Education)[:\s]*(.+?)(?:$|CONTACT|SKILLS)', combined, regex.IGNORECASE)
                if edu_match:
                    edu_text = edu_match.group(1).strip()
                    edu_key = edu_text.lower()[:50]  # Use first 50 chars as key
                    if edu_text and len(edu_text) > 10 and edu_key not in seen_education:
                        seen_education.add(edu_key)
                        result['education'].append(edu_text)
                
                # Also check for degree patterns - with deduplication
                degree_match = regex.search(r'(Masters?|Bachelor|B\.?Tech|M\.?Tech|MBA|MCA|BCA|B\.?E|M\.?E|Ph\.?D)[^,]*(?:in|of)[^,]+(?:from|at)[^,]+(?:University|College|Institute)[^,]*', combined, regex.IGNORECASE)
                if degree_match:
                    edu_text = degree_match.group(0).strip()
                    edu_key = edu_text.lower()[:50]
                    if edu_text and edu_key not in seen_education:
                        seen_education.add(edu_key)
                        result['education'].append(edu_text)
                
                # Extract certifications (GCP, AWS, Azure, etc.)
                if 'Certifications' in combined or 'GCP' in combined or 'AWS' in combined:
                    cert_matches = regex.findall(r'(GCP[^,\n]+|AWS[^,\n]+|Azure[^,\n]+|DP-?\d+[^,\n]*)', combined, regex.IGNORECASE)
                    for cert in cert_matches:
                        cert = cert.strip()
                        if cert and cert not in result['certifications']:
                            result['certifications'].append(cert)
    except Exception:
        pass
    
    return result


def extract_text_from_textboxes(file_path: str) -> str:
    """
    Extract text from DOCX text boxes (shapes).
    Used for multi-column layouts like Nageswara's resume where
    contact info is in a sidebar text box.
    """
    from docx import Document
    doc = Document(file_path)
    
    all_textbox_content = []
    
    # Find all w:txbxContent elements (text box content)
    for txbx in doc.element.iter():
        if txbx.tag.endswith('txbxContent'):
            texts = []
            for t in txbx.iter():
                if t.tag.endswith('}t') and t.text:
                    texts.append(t.text)
            if texts:
                content = ' '.join(texts)
                all_textbox_content.append(content)
    
    return '\n'.join(all_textbox_content)


def extract_all_text_from_docx(file_path: str) -> str:
    """
    Extract ALL text from DOCX: paragraphs + tables + text boxes.
    Handles multi-column layouts, sidebars, and complex formatting.
    """
    from docx import Document
    doc = Document(file_path)
    
    all_text = []
    
    # 1. Extract paragraphs
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            all_text.append(text)
    
    # 2. Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_text.append(cell_text)
            if row_text:
                all_text.append(' | '.join(row_text))
    
    # 3. Extract text boxes (for multi-column layouts)
    for txbx in doc.element.iter():
        if txbx.tag.endswith('txbxContent'):
            texts = []
            for t in txbx.iter():
                if t.tag.endswith('}t') and t.text:
                    texts.append(t.text)
            if texts:
                content = ' '.join(texts)
                # Avoid duplicates
                if content not in all_text:
                    all_text.append(content)
    
    return '\n'.join(all_text)


def extract_from_docx_tables(file_path: str) -> Dict:
    """
    Extract structured data from table-based DOCX resumes.
    Returns dict with: summary, skills, education, experience, responsibilities_tables
    """
    from docx import Document
    doc = Document(file_path)
    
    result = {
        'summary': '',
        'skills': {},
        'education': [],
        'experience': [],
        'tools': [],
        'responsibilities_tables': []  # For Nageswara-style resumes
    }
    
    for table_idx, table in enumerate(doc.tables):
        rows = table.rows
        if not rows:
            continue
        
        # Check first cell to determine table type
        first_cell = rows[0].cells[0].text.strip().lower() if rows[0].cells else ""
        first_cell_full = rows[0].cells[0].text.strip() if rows[0].cells else ""
        
        # Skills table: "EtlTools | Informatica..." or "TECHNICAL"
        if any(kw in first_cell for kw in ['tools', 'database', 'language', 'reporting', 'scheduling', 'technical', 'operating']):
            for row in rows:
                if len(row.cells) >= 2:
                    key = row.cells[0].text.strip()
                    value = row.cells[1].text.strip()
                    if key and value:
                        result['skills'][key] = value
        
        # Education table
        elif 'education' in first_cell:
            for row in rows:
                if len(row.cells) >= 2:
                    edu_text = row.cells[1].text.strip()
                    if edu_text and 'education' not in edu_text.lower():
                        result['education'].append(edu_text)
        
        # Experience table: "Client: X" pattern (Ramaswamy style)
        elif first_cell.startswith('client'):
            exp_entry = {
                'employer': '',
                'title': '',
                'duration': '',
                'description': '',
                'responsibilities': [],
                'tools': []
            }
            
            for row in rows:
                if not row.cells:
                    continue
                    
                cell0 = row.cells[0].text.strip().lower()
                # Get value from cell 1 or 2 (some tables have merged cells)
                cell_value = ''
                for cell in row.cells[1:]:
                    if cell.text.strip():
                        cell_value = cell.text.strip()
                        break
                
                if 'client' in cell0:
                    # "Client: UOB Bank" in first cell
                    client_text = row.cells[0].text.strip()
                    match = re.search(r'Client:\s*(.+)', client_text, re.IGNORECASE)
                    if match:
                        exp_entry['employer'] = match.group(1).strip()
                    
                    # Duration might be in another cell
                    for cell in row.cells[1:]:
                        dur_match = re.search(r'Duration:\s*(.+)', cell.text, re.IGNORECASE)
                        if dur_match:
                            exp_entry['duration'] = dur_match.group(1).strip()
                
                elif 'duration' in cell0:
                    dur_match = re.search(r'Duration:\s*(.+)', row.cells[0].text, re.IGNORECASE)
                    if dur_match:
                        exp_entry['duration'] = dur_match.group(1).strip()
                    elif cell_value:
                        exp_entry['duration'] = cell_value
                
                elif 'description' in cell0 or 'brief' in cell0:
                    exp_entry['description'] = cell_value
                
                elif 'responsibilities' in cell0 or 'role' in cell0:
                    # Split responsibilities by newline or bullet
                    resp_text = cell_value
                    resps = re.split(r'\n|•|►|■', resp_text)
                    exp_entry['responsibilities'] = [r.strip() for r in resps if r.strip() and len(r.strip()) > 20]
                
                elif 'tools' in cell0 or 'technologies' in cell0 or 'environment' in cell0:
                    exp_entry['tools'] = [t.strip() for t in re.split(r'[,|]', cell_value) if t.strip()]
            
            if exp_entry['employer']:
                result['experience'].append(exp_entry)
        
        # Responsibilities table (Nageswara style) - tables with responsibilities/environment
        elif 'environment' in first_cell or any(kw in first_cell_full for kw in ['responsible', 'worked on', 'team lead', 'developer', 'engineer']):
            resp_entry = {
                'responsibilities': [],
                'tools': [],
                'environment': ''
            }
            
            for row in rows:
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if not cell_text:
                        continue
                    
                    # Check if it's an environment line
                    if cell_text.lower().startswith('environment'):
                        resp_entry['environment'] = cell_text
                        # Extract tools from environment
                        tools_match = re.search(r'Environment\s*:\s*(.+)', cell_text, re.IGNORECASE)
                        if tools_match:
                            resp_entry['tools'] = [t.strip() for t in re.split(r'[,|]', tools_match.group(1)) if t.strip()]
                    else:
                        # It's responsibilities
                        resps = re.split(r'\n|•|►|■', cell_text)
                        for resp in resps:
                            resp = resp.strip()
                            if resp and len(resp) > 20 and not resp.lower().startswith('environment'):
                                resp_entry['responsibilities'].append(resp)
            
            if resp_entry['responsibilities'] or resp_entry['tools']:
                result['responsibilities_tables'].append(resp_entry)
        
        # Summary table (single cell with "years of experience")
        elif 'experience' in first_cell or 'year' in first_cell:
            result['summary'] = rows[0].cells[0].text.strip()
    
    return result


def clean_output_text(text: str) -> str:
    """Clean text for final output - remove artifacts and control characters."""
    if not text:
        return ""
    # Remove literal \n and actual newlines
    text = text.replace('\\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    # Remove any remaining control characters (ASCII 0-31 except space)
    text = ''.join(c if ord(c) >= 32 or c == ' ' else ' ' for c in text)
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
    
    # LinkedIn - multiple patterns (handle spaces from PDF extraction)
    linkedin_patterns = [
        r'linkedin\.com\s*/in/([\w-]+)',  # with optional space after .com
        r'linkedin\.com\s*/?\s*in/([\w-]+)',  # with optional spaces around /in
        r'linkedin[:\s]+(?:www\.)?linkedin\.com\s*/in/([\w-]+)',
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
        'experienced', 'skilled', 'dedicated', 'professional with', 'having',
        'languages', 'cloud', 'web applications', 'version', 'configuration',
        'operating', 'databases', 'tools', 'contact', 'phone', 'email', 'ph:',
        'work experience', 'work location', 'key skills', 'domain:',
        'data engineering', 'software engineering', 'governance', 'with over',
        'executive reporting', 'executive summary', 'total', 'years of'
    ]
    
    # Full patterns to skip (exact match after lowercase)
    skip_exact_patterns = [
        'technical skills', 'professional skills', 'key skills', 'core skills',
        'work experience', 'professional experience', 'contact information',
        'contact details', 'personal information', 'personal details'
    ]
    
    for line in lines:
        # Clean the line first (remove excessive whitespace)
        clean_line = ' '.join(line.split())
        
        # Remove markdown bold formatting: **name** or __name__ -> name
        clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_line)
        clean_line = re.sub(r'__([^_]+)__', r'\1', clean_line)
        
        # Remove contact info suffixes BEFORE checking skip patterns
        # "NAVEEN REDDY YADLA Contact: 470-505-9469" -> "NAVEEN REDDY YADLA"
        name_part = re.sub(r'\s*(Contact|Phone|Email|Tel|Cell|Mobile)[:\s].*$', '', clean_line, flags=re.IGNORECASE).strip()
        
        # Skip lines that START with skip patterns
        if any(name_part.lower().startswith(skip) for skip in skip_start_patterns):
            continue
        # Skip exact matches
        if name_part.lower() in skip_exact_patterns:
            continue
        # Skip lines ending with colon (likely headers)
        if name_part.endswith(':'):
            continue
        # Skip if entire line is an email or phone
        if re.match(r'^[\w.+-]+@[\w.-]+\.\w+$', name_part) or re.match(r'^[\d\s\-+()]+$', name_part):
            continue
        # Skip lines that look like skill categories
        if re.match(r'^[A-Za-z\s]+\s*:\s*', name_part):
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
                # Additional check: not tech terms or common headers
                tech_terms = ['SQL', 'ETL', 'GCP', 'AWS', 'API', 'XML', 'JSON', 'HTML', 'CSS', 'SSIS', 'SSAS']
                header_words = ['Technical', 'Skills', 'Professional', 'Experience', 'Summary', 'Contact', 'Information', 'Details', 'Work', 'Education', 'Objective']
                if not any(p in tech_terms for p in parts) and not all(p in header_words for p in parts):
                    if len(parts) == 2:
                        return parts[0], "", parts[1]
                    elif len(parts) == 3:
                        return parts[0], parts[1], parts[2]
                    else:
                        return parts[0], ' '.join(parts[1:-1]), parts[-1]
    
    # Fallback: Try to extract name from email address
    email_match = re.search(r'\b([a-zA-Z0-9._%+-]+)@', text)
    if email_match:
        email_local = email_match.group(1)
        # Try to parse email local part as name (e.g., "john.doe" -> "John Doe")
        # Skip if it looks like a generic email
        if not any(g in email_local.lower() for g in ['info', 'contact', 'admin', 'support', 'hr', 'sales']):
            # Split on dots, underscores, numbers
            name_parts = re.split(r'[._\d]+', email_local)
            name_parts = [p.capitalize() for p in name_parts if p and len(p) > 1]
            if len(name_parts) >= 2:
                return name_parts[0], "", name_parts[-1]
            elif len(name_parts) == 1:
                return name_parts[0], "", ""
    
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
    
    # Try to find "ROLE:" pattern in text
    role_match = re.search(r'ROLE[:\s]+([A-Za-z\s]+(?:Analyst|Engineer|Developer|Manager|Consultant|Lead|Specialist))', text, re.IGNORECASE)
    if role_match:
        return clean_output_text(role_match.group(1))
    
    # Fallback to most recent job title (cleaned)
    if experiences:
        for exp in experiences:
            title = exp.title
            if title:
                # Skip if title is "Work Location" or similar
                if re.match(r'^Work\s+Location', title, re.IGNORECASE):
                    continue
                if re.match(r'^ROLE:', title, re.IGNORECASE):
                    continue
                # Remove suffix like "– Cognizant Infra Services"
                title = re.sub(r'\s*[-–]\s*[A-Z][\w\s]+(?:Services|Solutions|Group|Team)?\s*$', '', title)
                # Remove location
                title = re.sub(r'\s+(Philadelphia|Chicago|Dallas|Plano|Bangalore|Bengaluru|Hyderabad|North Chicago|India|USA|Karnataka)[,\s]*(?:PA|TX|IL|NY|CA|USA|India)?$', '', title, flags=re.IGNORECASE)
                if title:
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
    date_pattern_mm = r'^\d{2}/\d{4}\s*[-–]'  # MM/YYYY format
    
    j = start_idx
    while j < len(lines) and j < start_idx + 30:
        resp_line = lines[j].strip()
        
        # Stop conditions
        if re.search(date_pattern, resp_line, re.IGNORECASE):
            break
        if re.match(date_pattern_mm, resp_line):
            break
        if re.match(r'^(EDUCATION|TECHNICAL|SKILLS|CERTIFICATIONS|PERSONAL|KEY\s+ARCH|PROFESSIONAL\s+EXPERIENCE)', resp_line, re.IGNORECASE):
            break
        if re.match(r'^[A-Z][A-Za-z\s&]+(?:Ltd|Inc|Corp|Technologies|Solutions)?\s*[-–]\s*[A-Z]', resp_line):
            break
        
        # Extract bullet points - include ■ (black square) used in some PDFs
        if resp_line.startswith(('•', '-', '*', '–', '■')) or re.match(r'^\d+\.', resp_line):
            resp_text = re.sub(r'^[•\-\*–■\d.]\s*', '', resp_line)
            if len(resp_text) > 20:
                responsibilities.append(clean_output_text(resp_text))
        # Also check for lines ending with ■ (some PDFs format this way)
        elif resp_line.endswith('■'):
            resp_text = resp_line.rstrip('■').strip()
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
    
    # Apostrophe format: Feb'20, May'21, Aug'15
    match = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)['\u2019](\d{2})", text, re.IGNORECASE)
    if match:
        month = MONTH_MAP.get(match.group(1).lower())
        year_short = int(match.group(2))
        year = 2000 + year_short if year_short < 50 else 1900 + year_short
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


def parse_table_duration(duration_str: str) -> Tuple[str, str, int]:
    """
    Parse duration string like "Nov-2023 to Till date" or "Sep-2022 to Oct-2023"
    Returns: (start_date, end_date, duration_months)
    """
    if not duration_str:
        return "", "", 0
    
    # Pattern: "Mon-YYYY to Mon-YYYY" or "Mon-YYYY to Till date"
    match = re.search(r'(\w{3})[-/]?(\d{4})\s+to\s+(\w+)[-/]?(\d{4})?', duration_str, re.IGNORECASE)
    if match:
        start_month_str = match.group(1).lower()
        start_year = int(match.group(2))
        end_str = match.group(3).lower()
        end_year_str = match.group(4)
        
        start_month = MONTH_MAP.get(start_month_str[:3], 1)
        
        if 'till' in end_str or 'present' in end_str or 'current' in end_str or 'date' in end_str:
            end_year = datetime.now().year
            end_month = datetime.now().month
        else:
            end_month = MONTH_MAP.get(end_str[:3], 12)
            end_year = int(end_year_str) if end_year_str else start_year
        
        duration = calculate_duration(start_year, start_month, end_year, end_month)
        start_date = f"{start_year}-{start_month:02d}"
        end_date = f"{end_year}-{end_month:02d}"
        
        return start_date, end_date, duration
    
    return "", "", 0


def extract_experiences_from_tables(table_data: Dict) -> List[ExperienceEntry]:
    """
    Convert table-extracted experience data to ExperienceEntry objects.
    """
    experiences = []
    
    for exp in table_data.get('experience', []):
        start_date, end_date, duration = parse_table_duration(exp.get('duration', ''))
        
        # Extract title from description or use generic
        title = ""
        desc = exp.get('description', '')
        if desc:
            # Try to find role in description
            role_match = re.search(r'(?:role|position|working as)\s*[:\-]?\s*([A-Za-z\s]+)', desc, re.IGNORECASE)
            if role_match:
                title = role_match.group(1).strip()
        
        if not title:
            # Try to infer from tools
            tools = exp.get('tools', [])
            if any('snowflake' in t.lower() for t in tools):
                title = "Data Engineer"
            elif any('informatica' in t.lower() for t in tools):
                title = "ETL Developer"
            elif any('teradata' in t.lower() for t in tools):
                title = "Data Warehouse Developer"
            else:
                title = "Data Professional"
        
        experiences.append(ExperienceEntry(
            employer=exp.get('employer', ''),
            title=title,
            location="",
            start_date=start_date,
            end_date=end_date,
            duration_months=duration,
            responsibilities=exp.get('responsibilities', [])[:12],
            tools=exp.get('tools', []),
            client=""
        ))
    
    return experiences


def extract_education_from_tables(table_data: Dict) -> List[Dict]:
    """
    Convert table-extracted education data to standard format.
    """
    education = []
    
    for edu_text in table_data.get('education', []):
        # Parse "Master of Computer Application (M.C.A) at Osmania University, India"
        degree = ""
        institution = ""
        year = None
        
        # Pattern: "Degree at/from Institution, Location"
        match = re.match(r'(.+?)\s+(?:at|from)\s+(.+)', edu_text, re.IGNORECASE)
        if match:
            degree = match.group(1).strip()
            institution = match.group(2).strip()
        else:
            degree = edu_text
        
        # Try to extract year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', edu_text)
        if year_match:
            year = year_match.group(1)
        
        education.append({
            'degree': degree,
            'institution': institution if institution else None,
            'year': year
        })
    
    return education
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
    
    # Strategy 0: "Title | Company, Location | Date Range" format (Steven style)
    # Example: "Sr.Cloud/DeVOPS Engineer | RElix, Tx | November 2022 -- Present"
    # Note: Uses [-–]+ to match single dash, en-dash, or double dash
    pipe_date_pattern = r'^([^|]+)\s*\|\s*([^|]+)\s*\|\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–]+\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current)$'
    for line in text.split('\n'):
        match = re.match(pipe_date_pattern, line.strip(), re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            company_loc = match.group(2).strip()
            start_str = match.group(3)
            end_str = match.group(4)
            
            # Parse company and location
            employer = company_loc
            location = ""
            if ',' in company_loc:
                parts = company_loc.rsplit(',', 1)
                employer = parts[0].strip()
                location = parts[1].strip() if len(parts) > 1 else ""
            
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
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        duration_months=duration
                    ))
    
    # Strategy 0a: "Title Company - Location MM/YYYY - MM/YYYY|Present" (Javvaji inline style)
    # Example: "Senior Java Developer LOWE'S - USA 05/2023 - Present"
    # Example: "Senior(Lead) Java Consultant Renault-Nissan - France & North America 02/2015 - 04/2023"
    inline_mm_pattern = r'^(.+?)\s+(\d{2}/\d{4})\s*[-–]\s*(\d{2}/\d{4}|Present|Current)\s*$'
    for line in text.split('\n'):
        match = re.match(inline_mm_pattern, line.strip(), re.IGNORECASE)
        if match:
            header = match.group(1).strip()
            start_str = match.group(2)
            end_str = match.group(3)
            
            # Skip section headers
            if header.upper() in ['EXPERIENCE', 'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT']:
                continue
            
            # Parse location from header
            location = ""
            loc_match = re.search(r'\s*[-–]\s*(USA|India|France|UK|Canada|Germany|North America|Europe|France & North America)\s*$', header, re.IGNORECASE)
            if loc_match:
                location = loc_match.group(1).strip()
                header = header[:loc_match.start()].strip()
            
            # Split title and company
            employer = ""
            title = header
            
            # Pattern: Look for uppercase company at end (LOWE'S, IBM, etc.)
            title_company_match = re.match(r'^(.+?)\s+([A-Z][A-Z\s&\']+(?:\'S)?)\s*$', header)
            if title_company_match:
                title = title_company_match.group(1).strip()
                employer = title_company_match.group(2).strip()
            else:
                # Pattern: Look for company like "Renault-Nissan" at end
                title_words = ['Developer', 'Engineer', 'Consultant', 'Manager', 'Analyst', 'Lead', 'Architect', 'Administrator']
                for tw in title_words:
                    idx = header.rfind(tw)
                    if idx > 0:
                        potential_title = header[:idx + len(tw)].strip()
                        potential_employer = header[idx + len(tw):].strip()
                        potential_employer = re.sub(r'^[\s\-–]+', '', potential_employer).strip()
                        if potential_employer:
                            title = potential_title
                            employer = potential_employer
                            break
            
            # Parse MM/YYYY dates
            start_parts = start_str.split('/')
            start_month = int(start_parts[0])
            start_year = int(start_parts[1])
            
            if end_str.lower() in ['present', 'current']:
                end_year = datetime.now().year
                end_month = datetime.now().month
                is_present = True
            else:
                end_parts = end_str.split('/')
                end_month = int(end_parts[0])
                end_year = int(end_parts[1])
                is_present = False
            
            duration = calculate_duration(start_year, start_month, end_year, end_month)
            start_date = f"{start_year}-{start_month:02d}"
            end_date = f"{end_year}-{end_month:02d}"
            
            is_dup = any(e.title == title and e.start_date == start_date for e in experiences)
            if not is_dup:
                # Get responsibilities from following lines
                lines = text.split('\n')
                line_idx = lines.index(line.strip()) if line.strip() in lines else -1
                responsibilities = []
                if line_idx >= 0:
                    responsibilities = extract_responsibilities(lines, line_idx + 1)
                
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
    
    # Strategy 0b: "Title Company - Location" then "Date Range" on next line (Javvaji style)
    # Example: "Senior Java Developer LOWE'S - USA" then "05/2023 - Present"
    # Also: "Senior(Lead) Java Consultant Renault-Nissan - France & North America" then "02/2015 - 04/2023"
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Check if next line is a date range
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            date_only_match = re.match(r'^(\d{2}/\d{4})\s*[-–]\s*(\d{2}/\d{4}|Present|Current)$', next_line, re.IGNORECASE)
            if date_only_match:
                # This line should be "Title Company - Location"
                header = line.strip()
                # Skip if it's a section header
                if header.upper() in ['EXPERIENCE', 'WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'EMPLOYMENT']:
                    continue
                
                # Parse: "Senior Java Developer LOWE'S - USA" or "Senior(Lead) Java Consultant Renault-Nissan - France & North America"
                loc_match = re.search(r'\s*[-–]\s*(USA|India|France|UK|Canada|Germany|North America|Europe|France & North America)\s*$', header, re.IGNORECASE)
                location = ""
                if loc_match:
                    location = loc_match.group(1).strip()
                    header = header[:loc_match.start()].strip()
                
                # Try to split title and company
                employer = ""
                title = header
                
                # Pattern 1: Look for uppercase company at end (LOWE'S, IBM, etc.)
                title_company_match = re.match(r'^(.+?)\s+([A-Z][A-Z\s&\']+(?:\'S)?)\s*$', header)
                if title_company_match:
                    title = title_company_match.group(1).strip()
                    employer = title_company_match.group(2).strip()
                else:
                    # Pattern 2: Look for company name like "Renault-Nissan" at end
                    # Title words: Developer, Engineer, Consultant, Manager, Analyst, Lead, etc.
                    title_words = ['Developer', 'Engineer', 'Consultant', 'Manager', 'Analyst', 'Lead', 'Architect', 'Administrator', 'Specialist', 'Director']
                    for tw in title_words:
                        idx = header.rfind(tw)
                        if idx > 0:
                            potential_title = header[:idx + len(tw)].strip()
                            potential_employer = header[idx + len(tw):].strip()
                            # Clean up employer
                            potential_employer = re.sub(r'^[\s\-–]+', '', potential_employer).strip()
                            if potential_employer:
                                title = potential_title
                                employer = potential_employer
                                break
                
                # Parse dates
                start_str = date_only_match.group(1)
                end_str = date_only_match.group(2)
                
                # Parse MM/YYYY format
                start_match = re.match(r'(\d{2})/(\d{4})', start_str)
                if start_match:
                    start_month = int(start_match.group(1))
                    start_year = int(start_match.group(2))
                else:
                    continue
                
                if 'present' in end_str.lower() or 'current' in end_str.lower():
                    end_year = datetime.now().year
                    end_month = datetime.now().month
                    is_present = True
                else:
                    end_match = re.match(r'(\d{2})/(\d{4})', end_str)
                    if end_match:
                        end_month = int(end_match.group(1))
                        end_year = int(end_match.group(2))
                        is_present = False
                    else:
                        continue
                
                duration = calculate_duration(start_year, start_month, end_year, end_month)
                start_date = f"{start_year}-{start_month:02d}"
                end_date = f"{end_year}-{end_month:02d}"
                
                # Get responsibilities (lines starting OR ending with bullets after the date)
                responsibilities = []
                j = i + 2
                while j < len(lines) and j < i + 30:
                    resp_line = lines[j].strip()
                    # Stop at next job header or section
                    if re.match(r'^\d{2}/\d{4}\s*[-–]', resp_line):
                        break
                    if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z]', resp_line):  # New title
                        break
                    if re.match(r'^(Senior|Junior|Lead)\s+', resp_line, re.IGNORECASE):  # New job
                        break
                    # Lines starting with bullet
                    if resp_line.startswith(('■', '•', '-', '*')):
                        resp = re.sub(r'^[■•\-\*]\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(clean_output_text(resp))
                    # Lines ending with bullet (Javvaji PDF style)
                    elif resp_line.endswith('■'):
                        resp = resp_line.rstrip('■').strip()
                        if len(resp) > 20:
                            responsibilities.append(clean_output_text(resp))
                    j += 1
                
                is_dup = any(e.employer == employer and e.start_date == start_date for e in experiences)
                if not is_dup and (title or employer):
                    experiences.append(ExperienceEntry(
                        employer=employer,
                        title=title,
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        duration_months=duration,
                        responsibilities=responsibilities[:12]
                    ))
    
    # Strategy 0c: "Company Date" then "Work Location" then "ROLE: Title" (Sarwar style)
    # Example: "BNP Paribas May 2021 – Aug 2024" then "Work Location – Bengaluru" then "ROLE: Business Analyst"
    date_range_line_pattern = r'^([A-Z][A-Za-z\s&]+)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4})\s*[-–]\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}|Present|Current)$'
    for i, line in enumerate(lines):
        match = re.match(date_range_line_pattern, line.strip(), re.IGNORECASE)
        if match:
            employer = match.group(1).strip()
            start_str = match.group(2)
            end_str = match.group(3)
            
            # Skip if employer looks like a section header
            if employer.upper() in ['WORK EXPERIENCE', 'PROFESSIONAL EXPERIENCE', 'EXPERIENCE', 'EMPLOYMENT']:
                continue
            
            start_year, start_month, _ = parse_date(start_str)
            end_year, end_month, is_present = parse_date(end_str)
            
            if not start_year or not end_year:
                continue
            
            # Look for "Work Location" and "ROLE:" in next few lines
            title = ""
            location = ""
            responsibilities = []
            
            for j in range(i + 1, min(i + 5, len(lines))):
                check_line = lines[j].strip()
                
                # Check for Work Location
                loc_match = re.match(r'^Work\s+Location\s*[-–]\s*(.+)$', check_line, re.IGNORECASE)
                if loc_match:
                    location = loc_match.group(1).strip()
                    continue
                
                # Check for ROLE: or DESIGNATION:
                role_match = re.match(r'^(?:ROLE|DESIGNATION)[:\s]+(.+)$', check_line, re.IGNORECASE)
                if role_match:
                    title = role_match.group(1).strip()
                    break
            
            # Get responsibilities (bullet points OR plain text after ROLE line)
            if title:
                in_responsibilities = False
                for j in range(i + 1, min(i + 50, len(lines))):
                    resp_line = lines[j].strip()
                    # Stop at next company header
                    if re.match(date_range_line_pattern, resp_line, re.IGNORECASE):
                        break
                    if resp_line.startswith('---'):
                        break
                    # Check if we've entered responsibilities section
                    if 'responsibilities' in resp_line.lower() and resp_line.endswith(':'):
                        in_responsibilities = True
                        continue
                    # Extract bullet points
                    if resp_line.startswith(('•', '-', '*')):
                        resp = re.sub(r'^[•\-\*]\s*', '', resp_line)
                        if len(resp) > 20:
                            responsibilities.append(clean_output_text(resp))
                    # Extract plain text lines in responsibilities section
                    elif in_responsibilities and len(resp_line) > 30 and resp_line[0].isupper():
                        # Skip section headers
                        if not resp_line.startswith(('Work Location', 'ROLE:', 'DESIGNATION:')):
                            responsibilities.append(clean_output_text(resp_line))
            
            duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
            start_date = f"{start_year}-{(start_month or 1):02d}"
            end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
            
            is_dup = any(e.employer == employer and e.start_date == start_date for e in experiences)
            if not is_dup and employer:
                experiences.append(ExperienceEntry(
                    employer=employer,
                    title=title if title else "Professional",
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    duration_months=duration,
                    responsibilities=responsibilities[:12]
                ))
    
    # Strategy 0d: "Title" then "Company | Account | Location | YYYY - Present/YYYY" (Chaitu style)
    # Handles multiple pipe variations:
    # - 3 pipes (4 sections): "Tech Mahindra | UPS Account | Alpharetta, GA | 2024 – Present"
    # - 2 pipes (3 sections): "Samsung Electronics | Plano, TX | 2020 – 2024"
    # - 2 pipes (3 sections): "TCS | Multiple Fortune 500 Clients | 2005 – 2012"
    
    # Pattern for 3 pipes (Company | Account | Location | Date)
    pipe3_pattern = r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(\d{4})\s*[-–]\s*(\d{4}|Present|Current)\s*$'
    # Pattern for 2 pipes (Company | Location/Clients | Date)
    pipe2_pattern = r'^(.+?)\s*\|\s*(.+?)\s*\|\s*(\d{4})\s*[-–]\s*(\d{4}|Present|Current)\s*$'
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        employer = ""
        client_or_account = ""
        location = ""
        start_year = None
        end_str = ""
        
        # Try 3-pipe pattern first
        match3 = re.match(pipe3_pattern, line_stripped, re.IGNORECASE)
        if match3:
            employer = match3.group(1).strip()
            client_or_account = match3.group(2).strip()
            location = match3.group(3).strip()
            start_year = int(match3.group(4))
            end_str = match3.group(5)
        else:
            # Try 2-pipe pattern
            match2 = re.match(pipe2_pattern, line_stripped, re.IGNORECASE)
            if match2:
                employer = match2.group(1).strip()
                loc_or_client = match2.group(2).strip()
                start_year = int(match2.group(3))
                end_str = match2.group(4)
                
                # Determine if second part is location or client
                # Location: contains state abbrev or city pattern
                if re.search(r',\s*[A-Z]{2}\b', loc_or_client) or re.search(r'\b[A-Z][a-z]+\s*,\s*[A-Z]{2}\b', loc_or_client):
                    location = loc_or_client
                else:
                    # Treat as client/description
                    client_or_account = loc_or_client
        
        if not start_year:
            continue
        
        # Skip if employer is empty or looks like education
        if not employer or employer.lower().startswith(('executive', 'bachelor', 'master', 'mba')):
            continue
        
        # Get title from previous line
        title = ""
        if i > 0:
            prev_line = lines[i - 1].strip()
            # Make sure it's not a section header, bullet point, or another pipe line
            if prev_line and not prev_line.startswith(('•', '▪', '-', '*', 'PROFESSIONAL', 'WORK', 'EXPERIENCE', 'AREAS')):
                if '|' not in prev_line and not re.search(r'\d{4}\s*[-–]', prev_line):
                    # Skip if it's a responsibility/bullet continuation
                    if not any(prev_line.lower().startswith(w) for w in ['ensuring', 'targeting', 'serving', 'driving']):
                        title = prev_line
        
        # Parse end year
        if end_str.lower() in ['present', 'current']:
            end_year = datetime.now().year
            end_month = datetime.now().month
            is_present = True
        else:
            end_year = int(end_str)
            end_month = 12
            is_present = False
        
        # Calculate duration (use January for start, December/current for end)
        duration = calculate_duration(start_year, 1, end_year, end_month)
        start_date = f"{start_year}-01"
        end_date = f"{end_year}-{end_month:02d}"
        
        # Get responsibilities (lines starting with ▪ or •)
        responsibilities = []
        j = i + 1
        while j < len(lines) and j < i + 30:
            resp_line = lines[j].strip()
            # Stop at next job header (pipe with year pattern)
            if re.search(r'\|\s*\d{4}\s*[-–]', resp_line):
                break
            # Stop at section headers
            if resp_line.upper().startswith(('EDUCATION', 'CERTIFICATIONS', 'TECHNICAL', 'SKILLS', 'AREAS OF')):
                break
            # Extract bullet points (▪, •, -, *)
            if resp_line.startswith(('▪', '•', '-', '*')):
                resp = re.sub(r'^[▪•\-\*]\s*', '', resp_line)
                if len(resp) > 20:
                    responsibilities.append(clean_output_text(resp))
            j += 1
        
        is_dup = any(e.employer == employer and e.start_date == start_date for e in experiences)
        if not is_dup and employer:
            experiences.append(ExperienceEntry(
                employer=employer,
                title=title if title else "Professional",
                location=location,
                start_date=start_date,
                end_date=end_date,
                duration_months=duration,
                responsibilities=responsibilities[:12],
                client=client_or_account if client_or_account else ""
            ))
    
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
    # Also handles: "Client: Hiscox Inc - Atlanta, Georgia Feb'20 to May'21"
    # IMPORTANT: Handle company names with hyphens like "Co-Op Financial"
    
    # Pattern: Look for "Client:" then find the LAST dash before a location pattern (City, State format)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if not line.strip().lower().startswith('client:'):
            continue
            
        # Find date at end of line (formats: Jul24, Jun21, Feb'20, Present, P)
        date_end_match = re.search(r"(\w{3}['\u2019]?\d{2})\s+to\s+(\w{3}['\u2019]?\d{2}|Present|P)\s*$", line.strip(), re.IGNORECASE)
        if not date_end_match:
            continue
        
        # Extract the part before dates
        before_date = line[:date_end_match.start()].strip()
        start_str = date_end_match.group(1)
        end_str = date_end_match.group(2)
        
        # Parse: "Client: Company – Location"
        # Find the LAST occurrence of " - " or " – " that's followed by a location pattern
        # Location patterns: "City, State", "City, State.", "City State"
        client_prefix = re.match(r'^Client:\s*', before_date, re.IGNORECASE)
        if not client_prefix:
            continue
        after_client = before_date[client_prefix.end():]
        
        # Look for location patterns: "City, State" or "City State" at the end
        loc_patterns = [
            r'\s+[-–]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,?\s*(?:GA|TX|CA|NY|IL|PA|OH|IA|FL|NC|VA|MA|NJ|WA|CO|AZ|TN|MO|MD|WI|MN|IN|OR|NV|UT|KS|AR|NE|NM|WV|ID|HI|ME|NH|RI|MT|DE|SD|ND|AK|VT|WY|DC|India|USA|Georgia|Texas|California)\.?)\s*$',
        ]
        
        employer = after_client.strip()
        location = ""
        
        for loc_pat in loc_patterns:
            loc_match = re.search(loc_pat, after_client, re.IGNORECASE)
            if loc_match:
                employer = after_client[:loc_match.start()].strip()
                location = loc_match.group(1).strip().rstrip('.')
                break
        
        # Parse dates
        start_year, start_month, _ = parse_date(start_str)
        end_year, end_month, is_present = parse_date(end_str)
        
        if not start_year or not end_year:
            continue
        
        # Next line should be title
        title = ""
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not next_line.startswith(('Client:', 'Environment:', 'Description:', 'Responsibilities:')):
                title = next_line
        
        # Get responsibilities
        responsibilities = []
        for j in range(i + 2, min(i + 40, len(lines))):
            resp_line = lines[j].strip()
            if resp_line.lower().startswith('client:') or resp_line.lower().startswith('environment:'):
                break
            if resp_line.startswith(('•', '-', '*', '■')) or (resp_line and not resp_line.startswith(('Description:', 'Responsibilities:'))):
                resp = re.sub(r'^[•\-\*■]\s*', '', resp_line)
                if len(resp) > 20:
                    responsibilities.append(clean_output_text(resp))
        
        duration = calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
        start_date = f"{start_year}-{(start_month or 1):02d}"
        end_date = f"{end_year}-{(end_month or 12):02d}" if not is_present else f"{datetime.now().year}-{datetime.now().month:02d}"
        
        is_dup = any(e.employer == employer and e.start_date == start_date for e in experiences)
        if not is_dup and employer:
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
    seen_degrees = set()  # Track unique degrees to avoid duplicates
    
    # Job titles to filter out - these should NOT be in education
    job_title_patterns = [
        r'specialist', r'manager', r'engineer', r'developer', r'analyst', r'consultant',
        r'director', r'lead', r'architect', r'administrator', r'coordinator', r'executive',
        r'officer', r'supervisor', r'head\s+of', r'vp\b', r'vice\s+president', r'delivery',
        r'global\s+business', r'professional\s+services', r'directed', r'led\s+', r'managed'
    ]
    
    def is_job_title(text: str) -> bool:
        """Check if text looks like a job title instead of education."""
        if not text:
            return False
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in job_title_patterns)
    
    # Find education section (multiple possible headers)
    # IMPORTANT: Require EDUCATION to be at the START of a line (section header)
    # This avoids matching "executive education" in the middle of text
    edu_match = re.search(
        r'^(?:EDUCATION(?:AL)?\s*(?:QUALIFICATION|BACKGROUND|DETAILS)?|ACADEMIC\s*(?:QUALIFICATION|BACKGROUND)?)[:\s]*\n(.+?)(?:\nROLES|\nPROFESSIONAL|\nWORK|\nTECHNICAL|\nPERSONAL|\nCERTIFI|\nCORE|\nAS\s+A\s+SCRUM|\nSKILLS|\nACHIEVEMENT|\nTOOLS|\n_+|\Z)',
        text, re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    
    # Also try inline patterns like "Masters in X from Y University"
    # Only match clear education patterns with institution names
    inline_patterns = [
        # "Masters in Computer Science from Sri Venkateswara University, Tirupati"
        r'(Masters?|Bachelor\'?s?|MBA|MCA|BCA|B\.?Tech|M\.?Tech|Ph\.?D|Doctorate)\s+(?:in|of)\s+([A-Za-z\s]{3,30}?)\s+from\s+([A-Za-z\s]{3,40}?(?:University|College|Institute)[A-Za-z\s,]{0,30})',
        # "Masters of Business Administration (Finance) - Osmania University"
        r'(Masters?|Bachelor\'?s?)\s+of\s+([A-Za-z\s()]+?)\s*[-–]\s*([A-Za-z\s]+(?:University|College|Institute))',
    ]
    
    for pattern in inline_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            degree_type = match.group(1).strip()
            field = match.group(2).strip()
            institution = match.group(3).strip()
            
            # Build unique key
            degree_str = f"{degree_type} of {field}".strip()
            degree_key = degree_str.lower()
            
            # Skip if already seen
            if degree_key in seen_degrees:
                continue
            seen_degrees.add(degree_key)
            
            education.append({
                'degree': degree_str,
                'institution': institution,
                'year': None
            })
    
    # Single-line "Education: Degree at/from Institution" format (Ramaswamy style)
    # Example: "Education: Master of Computer Application (M.C.A) at Osmania University, India"
    single_line_edu = re.search(
        r'Education[:\s]+([A-Za-z]+(?:\'s)?\s+of\s+[A-Za-z\s()\.]+)\s+(?:at|from)\s+([A-Za-z\s]+(?:University|College|Institute)[A-Za-z\s,]*)',
        text, re.IGNORECASE
    )
    if single_line_edu:
        degree = single_line_edu.group(1).strip()
        institution = single_line_edu.group(2).strip()
        degree_key = degree.lower()
        if degree_key not in seen_degrees:
            seen_degrees.add(degree_key)
            education.append({
                'degree': degree,
                'institution': institution,
                'year': None
            })
    
    # Fallback: Look for "University...Date" pattern followed by degree line (Khaliq format)
    # This handles resumes without an EDUCATION header
    if not edu_match and not education:
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Pattern: "University of X Date - Date" or "X University Date - Date"
            uni_match = re.search(r'((?:\w+\s+)?University(?:\s+of\s+\w+)?|(?:\w+\s+)?College|(?:\w+\s+)?Institute)[,\s]+(?:\w+[,\s]+)*(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4})', line, re.IGNORECASE)
            if uni_match:
                institution = line[:uni_match.end()].strip()
                # Clean institution - remove date part
                institution = re.sub(r'\s+\w+\s+\d{4}\s*[-–]\s*\w+\s+\d{4}.*$', '', institution).strip()
                year = uni_match.group(3).split()[-1]  # End year
                
                # Next line should be degree
                degree = None
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    degree_keywords = ['mba', 'bachelor', 'master', 'b.tech', 'm.tech', 'mca', 'bca', 'engineering', 'science', 'arts', 'commerce', 'computer']
                    if any(kw in next_line.lower() for kw in degree_keywords):
                        degree = next_line
                
                if institution and degree:
                    degree_key = degree.lower()
                    if degree_key not in seen_degrees:
                        seen_degrees.add(degree_key)
                        education.append({
                            'degree': degree,
                            'institution': institution,
                            'year': year
                        })
    
    if not edu_match and not education:
        return education
    
    if edu_match:
        edu_section = edu_match.group(1)
        lines = [l.strip() for l in edu_section.split('\n') if l.strip() and not l.strip().startswith('_')]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip irrelevant lines
            if re.match(r'^(Worked|Working|•|-|\*|PROFESSIONAL|Roles|As\s+a\s+Scrum|Facilitate|Client:|Role|Analysis|Find|Set up|Duration)', line, re.IGNORECASE):
                i += 1
                continue
            
            # CRITICAL: Skip lines that look like job/company headers (pipe with year range AND location)
            # Example to skip: "Samsung Electronics | Plano, TX | 2020 - 2024"
            # Example to keep: "Master of Computer Applications (MCA) - IGNOU | 2001-2004"
            # The difference: job headers have pipe BEFORE the year with location pattern
            if re.search(r'\|\s*[A-Za-z]+\s*,\s*[A-Z]{2}\s*\|\s*\d{4}\s*[-–]', line, re.IGNORECASE):
                i += 1
                continue
            # Also skip if it looks like: "Company | Description | YYYY - Present" (without location but with Present)
            if re.search(r'\|\s*\d{4}\s*[-–]\s*(?:Present|Current)', line, re.IGNORECASE):
                i += 1
                continue
            
            # Skip lines that are extra details about education (Specializations, Assistant VP, etc.)
            if line.lower().startswith(('specialization', 'assistant', 'dean', 'scholar', 'president', 'vice')):
                i += 1
                continue
            
            entry = {}
            
            # Format 1a: "Degree | Date | Institution" (Steven format)
            # Example: "Masters in computer science |December 2015 |Northwestern Polytechnic University"
            if line.count('|') >= 2:
                parts = [p.strip() for p in line.split('|')]
                
                # Check if middle part looks like a date (month/year)
                middle_is_date = bool(re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s*\d{4}|\d{4}', parts[1], re.IGNORECASE))
                
                if middle_is_date and len(parts) >= 3:
                    # Format: Degree | Date | Institution
                    entry['degree'] = parts[0]
                    entry['institution'] = parts[2] if len(parts) > 2 else None
                    year_match = re.search(r'(\d{4})', parts[1])
                    if year_match:
                        entry['year'] = year_match.group(1)
                else:
                    # Format: Degree | Institution | Date
                    entry['degree'] = parts[0]
                    entry['institution'] = parts[1]
                    year_match = re.search(r'(\d{4})', parts[-1])
                    if year_match:
                        entry['year'] = year_match.group(1)
                
                # Skip if degree looks like a job title
                if is_job_title(entry.get('degree', '')) or is_job_title(entry.get('institution', '')):
                    i += 1
                    continue
                
                # Check for duplicate
                degree_key = entry['degree'].lower()
                if degree_key not in seen_degrees:
                    seen_degrees.add(degree_key)
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
                
                # Skip if degree looks like a job title
                if is_job_title(entry['degree']) or is_job_title(entry.get('institution', '')):
                    i += 1
                    continue
                
                degree_key = entry['degree'].lower()
                if degree_key not in seen_degrees:
                    seen_degrees.add(degree_key)
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
                    
                    # Skip if institution looks like a job title
                    if is_job_title(entry.get('institution', '')) or is_job_title(entry.get('degree', '')):
                        i += 1
                        continue
                    
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
            # Use regex for word boundaries to avoid false positives like "must be present"
            degree_patterns = [
                r'\bmaster', r'\bbachelor', r'\bmba\b', r'\bmca\b', r'\bbca\b', 
                r'\bb\.?tech\b', r'\bm\.?tech\b', r'\bb\.?e\.?\b', r'\bm\.?e\.?\b', 
                r'\bph\.?d\b', r'^me\s', r'^be\s', r'\bengineering\b'
            ]
            if any(re.search(p, line.lower()) for p in degree_patterns):
                # Additional check: skip if line looks like responsibilities
                skip_patterns = [r'analyze', r'provide', r'support', r'data must', r'tools\s*[&:]', r'environment']
                if any(re.search(p, line.lower()) for p in skip_patterns):
                    i += 1
                    continue
                
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
                    # Final sanity check: degree should contain a degree keyword
                    degree_lower = entry['degree'].lower()
                    valid_degree = any(re.search(p, degree_lower) for p in [r'\bmaster', r'\bbachelor', r'\bmba\b', r'\bmca\b', r'\bbca\b', r'\btech\b', r'\bengineering\b', r'\bscience\b', r'\barts\b', r'\bcommerce\b', r'\bph\.?d'])
                    if valid_degree:
                        education.append(entry)
            
            i += 1
    
    # Ensure all entries have all fields (null if missing) and clean up
    cleaned = []
    for edu in education:
        degree = edu.get('degree') or None
        institution = edu.get('institution') or None
        
        # Clean leading pipe characters (Ramaswamy issue)
        if degree:
            degree = re.sub(r'^[\|\s]+', '', degree).strip()
        if institution:
            institution = re.sub(r'^[\|\s]+', '', institution).strip()
        
        # Skip if degree is empty after cleanup
        if not degree:
            continue
            
        cleaned.append({
            'degree': degree,
            'institution': institution if institution else None,
            'year': edu.get('year') or None
        })
    
    return cleaned


# ============================================================================
# CERTIFICATION EXTRACTION
# ============================================================================

def extract_certifications(text: str) -> List[str]:
    """Extract certifications."""
    certifications = []
    
    # Multiple section patterns - handle various headers
    # IMPORTANT: Stop at TECHNICAL header (TECHNICAL EXPERTISE, TECHNICAL SKILLS, etc.)
    cert_patterns = [
        r'CERTIFICATIONS?\s*[/&]?\s*(?:TRAININGS?)?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\nEDUCATION|\nSKILLS|\nTOOLS|\nWORK|\nTECHNICAL|\Z)',
        r'CERTIFICATES?\s*[/&]?\s*(?:TRAININGS?)?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\nTOOLS|\nTECHNICAL|\Z)',
        r'TRAININGS?\s*[/&]?\s*(?:CERTIFICATIONS?)?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\nSKILLS|\nTOOLS|\nTECHNICAL|\Z)',
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
                if re.match(r'^(PROFESSIONAL|EXPERIENCE|IMG|IBM|Cognizant|Dell|Wipro|Tools|Work|TECHNICAL|AI/ML|Platforms|Cloud|Data:)', line, re.IGNORECASE):
                    continue
                
                # Skip if line looks like technical skills (contains multiple commas with tech terms)
                if line.count(',') >= 3 and any(t in line.lower() for t in ['python', 'java', 'aws', 'gcp', 'azure', 'docker']):
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
    
    # =========================================================================
    # STEP 1: Enhanced DOCX extraction (tables + text boxes for multi-column)
    # =========================================================================
    table_data = None
    table_experiences = []
    table_education = []
    textbox_text = ""
    
    if params.file_path and params.file_path.endswith('.docx'):
        try:
            # Extract from tables
            table_data = extract_from_docx_tables(params.file_path)
            if table_data.get('experience'):
                table_experiences = extract_experiences_from_tables(table_data)
            if table_data.get('education'):
                table_education = extract_education_from_tables(table_data)
            
            # Extract from text boxes (for multi-column layouts like Nageswara)
            textbox_text = extract_text_from_textboxes(params.file_path)
            if textbox_text:
                # Merge textbox content with main text for contact/education extraction
                text = text + "\n" + textbox_text
        except Exception as e:
            pass  # Fall back to text extraction
    
    # =========================================================================
    # STEP 2: Standard text-based extraction
    # =========================================================================
    contact = extract_contact(text)
    firstname, middle, lastname = extract_name(text)
    experiences = extract_experiences(text)
    education = extract_education(text)
    certifications = extract_certifications(text)
    summary = extract_summary(text)
    title = extract_title(text, experiences)
    
    # =========================================================================
    # STEP 3: Merge table and text extractions (prefer table data if richer)
    # =========================================================================
    if table_experiences:
        # Use table experiences if they have more data
        table_resp_count = sum(len(e.responsibilities) for e in table_experiences)
        text_resp_count = sum(len(e.responsibilities) for e in experiences)
        
        if table_resp_count > text_resp_count or len(table_experiences) > len(experiences):
            experiences = table_experiences
    
    if table_education and (not education or len(table_education) > len(education)):
        education = table_education
    
    # Use table summary if available and text summary is empty
    if table_data and table_data.get('summary') and not summary:
        summary = table_data['summary']
    
    # =========================================================================
    # STEP 4: Merge responsibilities from tables (Nageswara style)
    # =========================================================================
    if table_data and table_data.get('responsibilities_tables'):
        resp_tables = table_data['responsibilities_tables']
        # Match responsibilities tables to experiences by order
        for i, exp in enumerate(experiences):
            if i < len(resp_tables) and (not exp.responsibilities or len(exp.responsibilities) == 0):
                exp.responsibilities = resp_tables[i].get('responsibilities', [])[:12]
                if not exp.tools and resp_tables[i].get('tools'):
                    exp.tools = resp_tables[i].get('tools', [])
    
    # =========================================================================
    # STEP 5: Merge textbox data (for multi-column/sidebar layouts like Nageswara)
    # =========================================================================
    if params.file_path and params.file_path.endswith('.docx'):
        try:
            textbox_data = extract_from_docx_textboxes(params.file_path)
            
            # Merge contact info from textboxes if missing
            if textbox_data.get('phone') and not contact.get('phone'):
                contact['phone'] = textbox_data['phone']
            if textbox_data.get('email') and not contact.get('email'):
                contact['email'] = textbox_data['email']
            if textbox_data.get('linkedin') and not contact.get('linkedin'):
                contact['linkedin'] = textbox_data['linkedin']
            
            # Merge education from textboxes if missing (with deduplication)
            if textbox_data.get('education') and not education:
                seen_edu_keys = set()
                for edu_text in textbox_data['education']:
                    # Skip duplicates
                    edu_key = edu_text.lower()[:50]
                    if edu_key in seen_edu_keys:
                        continue
                    seen_edu_keys.add(edu_key)
                    
                    # Parse the education text
                    degree_match = re.match(r'(.+?)\s+(?:from|at)\s+(.+)', edu_text, re.IGNORECASE)
                    if degree_match:
                        education.append({
                            'degree': degree_match.group(1).strip(),
                            'institution': degree_match.group(2).strip(),
                            'year': None
                        })
                    else:
                        education.append({
                            'degree': edu_text,
                            'institution': None,
                            'year': None
                        })
            
            # Merge certifications from textboxes
            if textbox_data.get('certifications'):
                for cert in textbox_data['certifications']:
                    if cert not in certifications:
                        certifications.append(cert)
        except Exception:
            pass
    
    # Apply data quality checks on experiences
    experiences = ensure_data_quality(experiences)
    
    # Build name
    name_parts = [firstname]
    if middle:
        name_parts.append(middle)
    name_parts.append(lastname)
    name = ' '.join(filter(None, name_parts))
    
    # Get location - prefer most recent job, fallback to contact
    # Validate location to avoid picking up tech terms
    def is_valid_location(loc: str) -> bool:
        if not loc:
            return False
        loc_lower = loc.lower()
        # Invalid if it contains tech terms
        tech_terms = ['git', 'ci', 'cd', 'sql', 'aws', 'gcp', 'azure', 'python', 'java', 
                      'docker', 'kubernetes', 'jenkins', 'powercenter', 'informatica',
                      'oracle', 'teradata', 'snowflake', 'spark', 'kafka', 'hadoop',
                      'linux', 'unix', 'windows', 'server', 'database', 'api', 'etl',
                      'hp', 'ibm', 'sap', 'mongodb', 'mysql', 'postgres', 'redis']
        if any(term in loc_lower for term in tech_terms):
            return False
        # Must contain at least one valid location indicator
        valid_indicators = ['usa', 'india', 'uk', 'canada', 'texas', 'tx', 'ca', 'ny', 
                           'il', 'pa', 'ga', 'nc', 'va', 'nj', 'ma', 'fl', 'oh', 'wa',
                           'chicago', 'dallas', 'plano', 'houston', 'atlanta', 'new york',
                           'bangalore', 'bengaluru', 'hyderabad', 'pune', 'mumbai', 'chennai',
                           'france', 'germany', 'london', 'singapore', 'remote']
        return any(ind in loc_lower for ind in valid_indicators)
    
    location = ""
    if experiences and experiences[0].location:
        if is_valid_location(experiences[0].location):
            location = experiences[0].location
    if not location:
        contact_loc = contact.get('location', '')
        if is_valid_location(contact_loc):
            location = contact_loc
    
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
            "education": [
                {
                    "degree": clean_output_text(e.get('degree', '')) if e.get('degree') else None,
                    "institution": clean_output_text(e.get('institution', '')) if e.get('institution') else None,
                    "year": e.get('year')
                }
                for e in education
            ],
            "certifications": [clean_output_text(c) for c in certifications],
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
