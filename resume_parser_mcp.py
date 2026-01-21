"""
Resume Parser MCP Server - Production Grade
============================================
Enterprise-level resume parsing with Google/LinkedIn quality standards.
"""

import json
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
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
    'experience': ['experience', 'work experience', 'employment', 'professional experience', 
                   'work history', 'career history', 'employment history'],
    'education': ['education', 'academic background', 'academic qualifications', 'degrees'],
    'skills': ['skills', 'technical skills', 'core competencies', 'expertise', 'technologies'],
    'summary': ['summary', 'professional summary', 'executive summary', 'profile', 'objective'],
    'certifications': ['certifications', 'certificates', 'licenses'],
}

SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "golang", 
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", 
        "bash", "shell", "powershell", "sql", "html", "css"
    ],
    "frontend_frameworks": [
        "react", "angular", "vue", "nextjs", "nuxt", "svelte", "jquery", "bootstrap",
        "tailwind", "material ui", "redux", "webpack", "vite"
    ],
    "backend_frameworks": [
        "django", "flask", "fastapi", "spring", "spring boot", "nodejs", "express",
        "nestjs", "rails", "asp.net", ".net", "laravel", "symfony"
    ],
    "cloud_platforms": [
        "aws", "amazon web services", "azure", "microsoft azure", "gcp", 
        "google cloud", "heroku", "digitalocean", "vercel", "netlify"
    ],
    "aws_services": [
        "ec2", "s3", "lambda", "rds", "dynamodb", "cloudfront", "route53",
        "ecs", "eks", "fargate", "sqs", "sns", "sagemaker", "cloudwatch"
    ],
    "databases": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "oracle", "sql server", "sqlite", "neo4j"
    ],
    "devops_tools": [
        "docker", "kubernetes", "k8s", "helm", "terraform", "ansible", "jenkins",
        "gitlab ci", "github actions", "circleci", "prometheus", "grafana",
        "datadog", "splunk", "nginx", "apache"
    ],
    "ai_ml": [
        "machine learning", "ml", "deep learning", "dl", "artificial intelligence", "ai",
        "neural networks", "nlp", "natural language processing", "computer vision",
        "generative ai", "llm", "large language models", "gpt", "bert",
        "langchain", "rag", "mlops", "recommendation systems"
    ],
    "ml_frameworks": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "hugging face", "transformers", "spacy", "opencv"
    ],
    "data_engineering": [
        "etl", "data pipelines", "data warehousing", "data modeling",
        "apache airflow", "airflow", "dagster", "prefect", "dbt",
        "apache spark", "spark", "pyspark", "apache kafka", "kafka",
        "apache flink", "data lake", "delta lake"
    ],
    "data_warehouses": [
        "snowflake", "bigquery", "redshift", "databricks", "synapse", "teradata"
    ],
    "business_intelligence": [
        "tableau", "power bi", "looker", "qlik", "metabase", "superset",
        "data visualization", "dashboards", "reporting"
    ],
    "methodologies": [
        "agile", "scrum", "kanban", "devops", "ci/cd", "microservices",
        "serverless", "domain driven design", "design patterns"
    ],
    "soft_skills": [
        "leadership", "team leadership", "people management", "mentoring",
        "communication", "presentation", "collaboration", "teamwork",
        "problem solving", "critical thinking", "strategic thinking",
        "project management", "stakeholder management"
    ]
}

ALL_SKILLS_FLAT: Set[str] = set()
for skills in SKILL_TAXONOMY.values():
    ALL_SKILLS_FLAT.update(skills)

INDUSTRY_DOMAINS = [
    "telecommunications", "telecom", "retail", "e-commerce", "ecommerce",
    "banking", "financial services", "finance", "fintech", "insurance",
    "healthcare", "pharma", "logistics", "supply chain", "manufacturing",
    "technology", "software", "media", "entertainment", "education"
]

SENIORITY_PATTERNS = {
    "c_level": [r'\b(?:chief|ceo|cto|cfo|coo|cio|ciso)\b'],
    "vp": [r'\b(?:vp|vice\s+president|svp|evp)\b'],
    "director": [r'\b(?:director|head\s+of|principal)\b'],
    "senior": [r'\b(?:senior|sr\.?|lead|staff)\s+\w+', r'\barchitect\b'],
    "mid": [r'\b(?:manager|specialist|analyst|engineer|developer)\b'],
    "junior": [r'\b(?:junior|jr\.?|associate|entry|intern)\b']
}


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ParsedDate:
    original: str
    year: int
    month: Optional[int] = None
    day: Optional[int] = None
    is_present: bool = False
    confidence: float = 0.0
    
    def to_datetime(self) -> datetime:
        if self.is_present:
            return datetime.now()
        return datetime(self.year, self.month or 1, self.day or 1)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['iso_date'] = self.to_datetime().isoformat()
        return d


@dataclass 
class DateRange:
    start: ParsedDate
    end: ParsedDate
    duration_months: int
    duration_years: float
    is_current: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_date': self.start.to_dict(),
            'end_date': self.end.to_dict(),
            'duration_months': self.duration_months,
            'duration_years': self.duration_years,
            'is_current': self.is_current
        }


@dataclass
class SkillMatch:
    name: str
    normalized: str
    category: str
    months_experience: int = 0
    years_experience: float = 0.0
    proficiency: str = "Unknown"
    confidence: float = 0.0
    contexts: List[str] = field(default_factory=list)


@dataclass
class ExperienceEntry:
    title: str
    company: str
    location: Optional[str]
    date_range: DateRange
    responsibilities: List[str]
    skills_used: List[str]
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title, 'company': self.company, 'location': self.location,
            'date_range': self.date_range.to_dict(),
            'responsibilities': self.responsibilities,
            'skills_used': self.skills_used, 'confidence': self.confidence
        }


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
    include_raw_sections: bool = Field(default=False)
    calculate_skill_durations: bool = Field(default=True)


class ExtractSkillsInput(BaseModel):
    text: str = Field(..., min_length=10)
    include_proficiency: bool = Field(default=True)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MatchJobInput(BaseModel):
    resume_json: Dict[str, Any]
    job_description: str = Field(..., min_length=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.JSON)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2022': '-', '\u00a0': ' ',
        '\r\n': '\n', '\r': '\n', '\t': ' '
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def normalize_skill_name(skill: str) -> str:
    skill = skill.lower().strip()
    skill = re.sub(r'[^\w\s\+\#\.]', ' ', skill)
    skill = re.sub(r'\s+', ' ', skill).strip()
    normalizations = {
        'react.js': 'react', 'reactjs': 'react', 'vue.js': 'vue', 'vuejs': 'vue',
        'node.js': 'nodejs', 'next.js': 'nextjs', 'express.js': 'express',
        'amazon web services': 'aws', 'google cloud platform': 'gcp',
        'microsoft azure': 'azure', 'k8s': 'kubernetes',
    }
    return normalizations.get(skill, skill)


# ============================================================================
# NAME PARSING
# ============================================================================

def parse_name(text: str) -> ParsedName:
    if not text:
        return ParsedName(full_name="", first_name="", last_name="", confidence=0.0)
    
    name = re.sub(r'\s+', ' ', text.strip())
    name = re.sub(r'[\|,].*$', '', name)
    name = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', name)
    name = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', name)
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
    
    compound_indicators = {'van', 'von', 'de', 'del', 'della', 'di', 'da', 
                          'la', 'le', 'mac', 'mc', "o'"}
    
    last_name_start = len(parts) - 1
    for i in range(1, len(parts) - 1):
        if parts[i].lower().rstrip('.') in compound_indicators:
            last_name_start = i
            break
    
    if last_name_start == len(parts) - 1:
        middle_name = ' '.join(parts[1:-1]) if len(parts) > 2 else None
        last_name = parts[-1]
    else:
        middle_name = ' '.join(parts[1:last_name_start]) if last_name_start > 1 else None
        last_name = ' '.join(parts[last_name_start:])
    
    return ParsedName(full_name=text, first_name=parts[0], middle_name=middle_name,
                     last_name=last_name, prefix=prefix, suffix=suffix, confidence=0.85)


# ============================================================================
# DATE PARSING
# ============================================================================

def parse_date_string(text: str) -> Optional[ParsedDate]:
    if not text:
        return None
    
    text = text.strip().lower()
    original = text
    
    present_patterns = ['present', 'current', 'now', 'ongoing', 'today']
    if any(p in text for p in present_patterns):
        now = datetime.now()
        return ParsedDate(original=original, year=now.year, month=now.month,
                         is_present=True, confidence=1.0)
    
    # Month Year pattern
    match = re.search(r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*[,.]?\s*(\d{4})', text)
    if match:
        month = MONTH_MAP.get(match.group(1)[:3].lower())
        year = int(match.group(2))
        if month and 1900 <= year <= 2100:
            return ParsedDate(original=original, year=year, month=month, confidence=0.95)
    
    # MM/YYYY pattern
    match = re.search(r'(\d{1,2})[/\-](\d{4})', text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return ParsedDate(original=original, year=year, month=month, confidence=0.9)
    
    # Just year
    match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if match:
        return ParsedDate(original=original, year=int(match.group(1)), confidence=0.7)
    
    return None


def parse_date_range(text: str) -> Optional[DateRange]:
    if not text:
        return None
    
    text = normalize_text(text)
    
    for sep in [' - ', ' to ', ' until ', '-']:
        if sep in text.lower():
            parts = re.split(re.escape(sep), text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                break
    else:
        return None
    
    start_date = parse_date_string(parts[0].strip())
    end_date = parse_date_string(parts[1].strip())
    
    if not start_date or not end_date:
        return None
    
    start_dt = start_date.to_datetime()
    end_dt = end_date.to_datetime()
    
    if end_dt < start_dt:
        start_date, end_date = end_date, start_date
        start_dt, end_dt = end_dt, start_dt
    
    delta = relativedelta(end_dt, start_dt)
    duration_months = delta.years * 12 + delta.months
    if duration_months == 0:
        duration_months = 1
    
    return DateRange(start=start_date, end=end_date, duration_months=duration_months,
                    duration_years=round(duration_months / 12, 1), is_current=end_date.is_present)


# ============================================================================
# SECTION DETECTION
# ============================================================================

def detect_sections(text: str) -> Dict[str, str]:
    lines = text.split('\n')
    sections = {}
    current_section = 'header'
    current_content = []
    
    for line in lines:
        line_lower = line.strip().lower()
        line_clean = re.sub(r'[^\w\s]', '', line_lower)
        
        found_section = None
        for section_type, headers in SECTION_HEADERS.items():
            for header in headers:
                if line_clean == header or (len(line_clean) < 50 and header in line_clean):
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

def extract_skills_advanced(text: str) -> Dict[str, List[SkillMatch]]:
    text_lower = text.lower()
    found_skills: Dict[str, List[SkillMatch]] = {}
    seen_normalized: Set[str] = set()
    
    for category, skills in SKILL_TAXONOMY.items():
        found_skills[category] = []
        
        for skill in skills:
            normalized = normalize_skill_name(skill)
            if normalized in seen_normalized:
                continue
            
            pattern = rf'\b{re.escape(skill)}(?:\.?js)?\b'
            matches = list(re.finditer(pattern, text_lower))
            
            if matches:
                seen_normalized.add(normalized)
                contexts = [text[max(0, m.start()-50):min(len(text), m.end()+50)] 
                           for m in matches[:3]]
                confidence = 0.9 if len(skill) > 3 else 0.7
                
                found_skills[category].append(SkillMatch(
                    name=skill, normalized=normalized, category=category,
                    confidence=confidence, contexts=contexts
                ))
    
    return {k: v for k, v in found_skills.items() if v}


def calculate_skill_durations(skills: Dict[str, List[SkillMatch]], 
                             experiences: List[ExperienceEntry]) -> Dict[str, List[SkillMatch]]:
    """
    Calculate experience duration for each skill based on work history.
    Uses multiple strategies:
    1. Direct mention in responsibilities
    2. Skills extracted from job entry
    3. Related technology inference
    """
    # Build skill map
    skill_map: Dict[str, SkillMatch] = {}
    for cat_skills in skills.values():
        for s in cat_skills:
            skill_map[s.normalized] = s
            skill_map[s.name.lower()] = s
    
    # Related skills - if one is used, likely has experience with related ones
    related_skills = {
        'spring boot': ['spring', 'java'],
        'spring': ['java'],
        'springboot': ['spring', 'java'],
        'react': ['javascript'],
        'angular': ['typescript', 'javascript'],
        'vue': ['javascript'],
        'nextjs': ['react', 'javascript'],
        'django': ['python'],
        'flask': ['python'],
        'fastapi': ['python'],
        'pytorch': ['python', 'machine learning'],
        'tensorflow': ['python', 'machine learning'],
        'keras': ['python', 'tensorflow'],
        'pandas': ['python'],
        'scikit-learn': ['python', 'machine learning'],
        'pyspark': ['python', 'spark'],
        'express': ['nodejs', 'javascript'],
        'nestjs': ['nodejs', 'typescript'],
        'rails': ['ruby'],
        'laravel': ['php'],
        'asp.net': ['.net', 'c#'],
        'kubernetes': ['docker'],
        'eks': ['kubernetes', 'aws'],
        'ecs': ['aws', 'docker'],
        'lambda': ['aws'],
        'sagemaker': ['aws', 'machine learning'],
        'bigquery': ['gcp', 'sql'],
        'azure functions': ['azure'],
        'mlops': ['machine learning'],
        'deep learning': ['machine learning'],
        'nlp': ['machine learning'],
        'computer vision': ['machine learning'],
    }
    
    # Track job assignments to avoid double counting
    skill_jobs: Dict[str, set] = {k: set() for k in skill_map}
    
    for job_idx, exp in enumerate(experiences):
        duration = exp.date_range.duration_months
        
        # Combine all text from this job
        all_text_parts = [
            exp.title or "",
            exp.company or "",
            " ".join(exp.responsibilities) if exp.responsibilities else "",
        ]
        all_text = " ".join(all_text_parts).lower()
        
        # Get skills already extracted for this job
        job_skills_found = set(s.lower() for s in exp.skills_used) if exp.skills_used else set()
        
        # Find skills in this job's text
        matched_in_job = set()
        
        for key, skill in skill_map.items():
            if job_idx in skill_jobs.get(key, set()):
                continue
                
            skill_lower = skill.name.lower()
            
            # Check if skill appears in job text or was already extracted
            found = False
            
            # Method 1: Check skills_used from extraction
            if skill_lower in job_skills_found or skill.normalized in job_skills_found:
                found = True
            
            # Method 2: Direct text search with word boundaries
            if not found:
                patterns = [
                    rf'\b{re.escape(skill_lower)}\b',
                    rf'\b{re.escape(skill.normalized)}\b',
                ]
                for pattern in patterns:
                    if re.search(pattern, all_text):
                        found = True
                        break
            
            # Method 3: Handle variations (e.g., "ML" for "machine learning")
            if not found and len(skill_lower) <= 3:
                # Short acronyms - be more careful
                pattern = rf'\b{re.escape(skill_lower)}\b'
                if re.search(pattern, all_text):
                    found = True
            
            if found:
                matched_in_job.add(key)
        
        # Add related skills
        related_to_add = set()
        for matched_key in matched_in_job:
            skill = skill_map.get(matched_key)
            if skill:
                skill_lower = skill.name.lower()
                if skill_lower in related_skills:
                    for related in related_skills[skill_lower]:
                        related_norm = normalize_skill_name(related)
                        if related_norm in skill_map:
                            related_to_add.add(related_norm)
                        elif related in skill_map:
                            related_to_add.add(related)
        
        matched_in_job.update(related_to_add)
        
        # Assign duration to all matched skills
        for skill_key in matched_in_job:
            if skill_key in skill_map and job_idx not in skill_jobs.get(skill_key, set()):
                skill = skill_map[skill_key]
                skill.months_experience += duration
                skill.years_experience = round(skill.months_experience / 12, 1)
                if skill_key not in skill_jobs:
                    skill_jobs[skill_key] = set()
                skill_jobs[skill_key].add(job_idx)
    
    # Set proficiency based on years
    for skill in skill_map.values():
        years = skill.years_experience
        if years >= 8:
            skill.proficiency = "Expert"
        elif years >= 5:
            skill.proficiency = "Advanced"
        elif years >= 3:
            skill.proficiency = "Intermediate"
        elif years >= 1:
            skill.proficiency = "Beginner"
        else:
            skill.proficiency = "Familiar"
    
    return skills


# ============================================================================
# EXPERIENCE EXTRACTION
# ============================================================================

def extract_experiences(text: str, experience_section: str = None) -> List[ExperienceEntry]:
    experiences = []
    
    if not experience_section:
        sections = detect_sections(text)
        experience_section = sections.get('experience', '')
    
    if not experience_section:
        return experiences
    
    date_pattern = r'((?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{4}|\d{1,2}[/\-]\d{4}|\d{4})\s*[-to]+\s*(present|current|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*\d{4}|\d{1,2}[/\-]\d{4}|\d{4})'
    
    date_matches = list(re.finditer(date_pattern, experience_section, re.IGNORECASE))
    
    for i, match in enumerate(date_matches):
        date_range = parse_date_range(match.group(0))
        if not date_range:
            continue
        
        prev_end = date_matches[i-1].end() if i > 0 else 0
        header_text = experience_section[prev_end:match.start()].strip()
        
        next_start = date_matches[i+1].start() if i < len(date_matches) - 1 else len(experience_section)
        body_text = experience_section[match.end():next_start].strip()
        
        title, company, location = parse_job_header(header_text)
        responsibilities = parse_responsibilities(body_text)
        
        if title or company:
            experiences.append(ExperienceEntry(
                title=title or "Unknown", company=company or "Unknown",
                location=location, date_range=date_range,
                responsibilities=responsibilities, skills_used=[],
                confidence=0.8 if title and company else 0.5
            ))
    
    return experiences


def parse_job_header(text: str) -> Tuple[str, str, Optional[str]]:
    lines = [l.strip() for l in text.split('\n') if l.strip()][-3:]
    if not lines:
        return "", "", None
    
    header = ' | '.join(lines)
    
    for sep in [' at ', ' @ ', ' - ', ' | ']:
        if sep in header.lower():
            parts = re.split(re.escape(sep), header, maxsplit=2, flags=re.IGNORECASE)
            if len(parts) >= 2:
                role_kw = ['engineer', 'developer', 'manager', 'director', 'lead', 'head', 'chief']
                p1_is_title = any(kw in parts[0].lower() for kw in role_kw)
                
                if p1_is_title:
                    return parts[0].strip(), parts[1].strip(), parts[2].strip() if len(parts) > 2 else None
                else:
                    return parts[1].strip(), parts[0].strip(), None
    
    return lines[0] if lines else "", "", None


def parse_responsibilities(text: str) -> List[str]:
    parts = re.split(r'(?:^|\n)\s*(?:[-*]|\d+[.\)])\s*', text)
    return [re.sub(r'\s+', ' ', p.strip()) for p in parts if len(p.strip()) > 20][:10]


def calculate_total_experience(experiences: List[ExperienceEntry]) -> int:
    """
    Calculate total months of experience without double-counting overlapping jobs.
    Uses interval merging algorithm.
    """
    if not experiences:
        return 0
    
    # Convert experiences to date intervals (year, month tuples)
    intervals = []
    for exp in experiences:
        start = exp.date_range.start
        end = exp.date_range.end
        
        start_val = start.year * 12 + (start.month or 1)
        end_val = end.year * 12 + (end.month or 12)
        
        if end_val >= start_val:
            intervals.append((start_val, end_val))
    
    if not intervals:
        return 0
    
    # Sort by start date
    intervals.sort(key=lambda x: x[0])
    
    # Merge overlapping intervals
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:  # Overlapping or adjacent
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    # Calculate total months from merged intervals
    total = sum(end - start + 1 for start, end in merged)
    
    return total


# ============================================================================
# MAIN PARSING
# ============================================================================

async def parse_resume_full(params: ParseResumeInput) -> str:
    text = normalize_text(params.resume_text)
    sections = detect_sections(text)
    
    header_lines = [l.strip() for l in sections.get('header', '').split('\n') if l.strip()]
    parsed_name = parse_name(header_lines[0] if header_lines else "")
    
    title = ""
    if len(header_lines) > 1:
        potential = header_lines[1]
        if not re.search(r'@|linkedin|github|\d{3}[-.\s]\d{3}', potential, re.IGNORECASE):
            title = potential
    
    contact = extract_contact(text)
    experiences = extract_experiences(text, sections.get('experience', ''))
    
    # Calculate total experience properly (handling overlapping jobs)
    total_months = calculate_total_experience(experiences)
    
    skills = extract_skills_advanced(text)
    if params.calculate_skill_durations and experiences:
        skills = calculate_skill_durations(skills, experiences)
    
    seniority = detect_seniority(title, total_months / 12)
    
    result = {
        'name': parsed_name.to_dict(),
        'title': title,
        'seniority': seniority,
        'contact': contact,
        'summary': sections.get('summary', '')[:1000].strip() or None,
        'total_experience': {'months': total_months, 'years': round(total_months / 12, 1)},
        'skills': {cat: [{'name': s.name, 'months_experience': s.months_experience,
                         'years_experience': s.years_experience, 'proficiency': s.proficiency,
                         'confidence': s.confidence} for s in lst] 
                  for cat, lst in skills.items()},
        'experience': [e.to_dict() for e in experiences],
        'education': extract_education(text),
        'industries': [i for i in INDUSTRY_DOMAINS if i in text.lower()],
        'metadata': {'parsed_at': datetime.now().isoformat(), 'version': '2.0.0'}
    }
    
    return json.dumps(result, indent=2, default=str)


def extract_contact(text: str) -> Dict[str, Optional[str]]:
    contact = {'email': None, 'phone': None, 'linkedin': None, 'github': None, 'location': None}
    
    email = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text)
    if email:
        contact['email'] = email.group(1).lower()
    
    phone = re.search(r'\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})', text)
    if phone:
        contact['phone'] = f"({phone.group(1)}) {phone.group(2)}-{phone.group(3)}"
    
    linkedin = re.search(r'linkedin\.com/in/([\w-]+)', text, re.IGNORECASE)
    if linkedin:
        contact['linkedin'] = f"linkedin.com/in/{linkedin.group(1)}"
    
    location = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z]{2})\b', text)
    if location:
        contact['location'] = f"{location.group(1)}, {location.group(2)}"
    
    return contact


def extract_education(text: str) -> List[Dict[str, Any]]:
    education = []
    sections = detect_sections(text)
    edu_text = sections.get('education', '')
    
    degree_patterns = [(r"(Ph\.?D\.?)", "PhD"), (r"(M\.?B\.?A\.?)", "MBA"),
                      (r"(M\.?S\.?|Master)", "MS"), (r"(B\.?S\.?|B\.?A\.?|Bachelor)", "BS")]
    
    for pattern, label in degree_patterns:
        if re.search(pattern, edu_text, re.IGNORECASE):
            entry = {'degree': label}
            uni = re.search(r'(University|Institute|College)[\w\s,]+', edu_text, re.IGNORECASE)
            if uni:
                entry['institution'] = uni.group(0).strip()
            year = re.search(r'\b(19\d{2}|20\d{2})\b', edu_text)
            if year:
                entry['year'] = year.group(1)
            education.append(entry)
            break
    
    return education


def detect_seniority(title: str, years: float) -> Dict[str, Any]:
    title_lower = title.lower() if title else ""
    
    for level, patterns in SENIORITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, title_lower):
                return {'level': level.replace('_', '-').title(), 'source': 'title', 'confidence': 0.9}
    
    level = ("C-Level" if years >= 20 else "VP" if years >= 15 else "Director" if years >= 10
             else "Senior" if years >= 5 else "Mid" if years >= 2 else "Junior")
    
    return {'level': level, 'source': 'experience', 'confidence': 0.7}


# ============================================================================
# MCP TOOLS
# ============================================================================

if MCP_AVAILABLE:
    @mcp.tool(name="resume_parse_full")
    async def resume_parse_full_tool(params: ParseResumeInput) -> str:
        """Parse a complete resume with production-grade extraction."""
        return await parse_resume_full(params)
    
    @mcp.tool(name="resume_extract_skills")
    async def resume_extract_skills(params: ExtractSkillsInput) -> str:
        """Extract and categorize skills from text."""
        skills = extract_skills_advanced(params.text)
        filtered = {cat: [{'name': s.name, 'confidence': s.confidence} 
                         for s in lst if s.confidence >= params.min_confidence]
                   for cat, lst in skills.items()}
        return json.dumps({'skills_by_category': {k: v for k, v in filtered.items() if v}}, indent=2)
    
    @mcp.tool(name="resume_parse_name")
    async def resume_parse_name(name_text: str) -> str:
        """Parse a name string into structured components."""
        return json.dumps(parse_name(name_text).to_dict(), indent=2)
    
    @mcp.tool(name="resume_parse_dates")
    async def resume_parse_dates(date_text: str) -> str:
        """Parse a date range string and calculate duration."""
        dr = parse_date_range(date_text)
        if not dr:
            return json.dumps({'error': 'Could not parse', 'input': date_text})
        return json.dumps(dr.to_dict(), indent=2)
    
    @mcp.tool(name="resume_match_job")
    async def resume_match_job(params: MatchJobInput) -> str:
        """Analyze how well a resume matches a job description."""
        job_skills = {s.normalized for lst in extract_skills_advanced(params.job_description).values() for s in lst}
        resume_skills = {normalize_skill_name(s.get('name', s) if isinstance(s, dict) else s)
                        for lst in params.resume_json.get('skills', {}).values() for s in lst}
        
        matched = job_skills & resume_skills
        missing = job_skills - resume_skills
        score = (len(matched) / len(job_skills) * 100) if job_skills else 0
        
        return json.dumps({
            'match_score': round(score, 1),
            'matched_skills': list(matched),
            'missing_skills': list(missing),
            'skill_coverage': f"{len(matched)}/{len(job_skills)}"
        }, indent=2)
    
    @mcp.tool(name="resume_validate")
    async def resume_validate(resume_json: Dict[str, Any]) -> str:
        """Validate parsed resume JSON."""
        issues, warnings = [], []
        
        name = resume_json.get('name', {})
        if not name.get('first_name'):
            issues.append("Missing first_name")
        
        for i, exp in enumerate(resume_json.get('experience', [])):
            if exp.get('date_range', {}).get('duration_months', 0) == 0:
                issues.append(f"Experience {i+1}: duration is 0")
        
        return json.dumps({'is_valid': len(issues) == 0, 'issues': issues, 'warnings': warnings}, indent=2)


if __name__ == "__main__":
    if MCP_AVAILABLE:
        mcp.run()
    else:
        print("MCP not available. Use api_server.py for REST API.")
