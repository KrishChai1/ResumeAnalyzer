"""
Intelligent Resume Parser v4.0 - Advanced Multi-Agent System
=============================================================
Enterprise-grade resume parsing with AI-first approach.

AGENTS:
1. EXTRACTION AGENT - AI-powered data extraction (Claude)
2. RULES AGENT - Pattern-based extraction (regex fallback)
3. VALIDATION AGENT - Quality scoring & data verification
4. ENHANCEMENT AGENT - Gap filling & normalization

Works with ANY resume format - no hardcoding required.
"""

import os
import re
import json
import asyncio
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

# ============================================================================
# CONFIGURATION
# ============================================================================

MONTH_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

# ============================================================================
# AGENT 1: EXTRACTION AGENT (AI-Powered)
# ============================================================================

class ExtractionAgent:
    """AI-powered extraction using Claude API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "claude-sonnet-4-20250514"
    
    async def extract(self, text: str) -> Dict:
        """Extract all resume data using Claude AI."""
        if not self.api_key:
            return {}
        
        prompt = self._build_prompt(text)
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 8000,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("content", [{}])[0].get("text", "")
                    return self._parse_response(content)
        except Exception as e:
            print(f"AI Extraction error: {e}")
        
        return {}
    
    def _build_prompt(self, text: str) -> str:
        return f"""You are an expert resume parser. Extract ALL information from this resume.

RESUME TEXT:
{text[:20000]}

Return ONLY valid JSON (no markdown, no explanation) with this EXACT structure:
{{
    "name": {{
        "full": "Full Name",
        "first": "First",
        "middle": "Middle or null",
        "last": "Last"
    }},
    "contact": {{
        "email": "email@example.com",
        "phone": "+1234567890",
        "linkedin": "linkedin.com/in/username or null",
        "location": "City, State/Country"
    }},
    "title": "Current Professional Title (e.g., 'Senior Software Engineer', 'Project Manager')",
    "summary": "Professional summary text",
    "experience": [
        {{
            "employer": "Company Name",
            "title": "Job Title",
            "location": "City, State",
            "start_date": "YYYY-MM",
            "end_date": "YYYY-MM or Present",
            "is_current": true/false,
            "responsibilities": ["Responsibility 1", "Responsibility 2"],
            "technologies": ["Tech1", "Tech2"]
        }}
    ],
    "education": [
        {{
            "degree": "Degree Name",
            "institution": "University Name",
            "year": "YYYY",
            "field": "Field of Study"
        }}
    ],
    "certifications": ["Cert 1", "Cert 2"],
    "skills": {{
        "technical": ["Skill1", "Skill2"],
        "tools": ["Tool1", "Tool2"],
        "domains": ["Domain1", "Domain2"]
    }}
}}

RULES:
1. Extract EVERY job - do not skip any
2. For dates: convert any format to YYYY-MM (e.g., "Jan 2020" → "2020-01", "2020" → "2020-01")
3. For "Present", "Current", "Till Date" → use "Present"
4. Extract ALL responsibilities as separate items
5. If information is missing, use null
6. Return ONLY the JSON object, nothing else"""

    def _parse_response(self, content: str) -> Dict:
        """Parse AI response to extract JSON."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {}


# ============================================================================
# AGENT 2: RULES AGENT (Pattern-Based Extraction)
# ============================================================================

class RulesAgent:
    """Pattern-based extraction using regex rules."""
    
    def extract(self, text: str) -> Dict:
        """Extract resume data using regex patterns."""
        return {
            "name": self._extract_name(text),
            "contact": self._extract_contact(text),
            "title": self._extract_title(text),
            "experience": self._extract_experience(text),
            "education": self._extract_education(text),
            "certifications": self._extract_certifications(text),
            "skills": self._extract_skills(text)
        }
    
    def _extract_name(self, text: str) -> Dict:
        """Extract name from resume."""
        lines = [l.strip() for l in text.split('\n') if l.strip()][:20]
        
        skip_patterns = [
            'resume', 'cv', 'curriculum', 'summary', 'objective', 'professional',
            'experience', 'education', 'skills', 'technical', 'contact'
        ]
        
        for line in lines:
            clean = re.sub(r'\s*(Contact|Phone|Email|Tel|Cell|Mobile)[:\s].*$', '', line, flags=re.IGNORECASE).strip()
            clean = re.sub(r'[\|].*$', '', clean).strip()
            clean = re.sub(r'\s*,.*$', '', clean).strip()
            
            if any(skip in clean.lower() for skip in skip_patterns):
                continue
            if re.match(r'^[\w.+-]+@[\w.-]+\.\w+$', clean):
                continue
            if re.match(r'^[\d\s\-+()]+$', clean):
                continue
            
            parts = clean.split()
            if 2 <= len(parts) <= 4 and all(p[0].isupper() for p in parts if p):
                tech_terms = ['SQL', 'ETL', 'GCP', 'AWS', 'API', 'XML', 'JSON']
                if not any(p in tech_terms for p in parts):
                    return {
                        "full": clean,
                        "first": parts[0],
                        "middle": ' '.join(parts[1:-1]) if len(parts) > 2 else None,
                        "last": parts[-1]
                    }
        
        return {"full": None, "first": None, "middle": None, "last": None}
    
    def _extract_contact(self, text: str) -> Dict:
        """Extract contact information."""
        contact = {"email": None, "phone": None, "linkedin": None, "location": None}
        
        # Email
        email_match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', text)
        if email_match:
            contact["email"] = email_match.group(1).lower()
        
        # Phone - multiple patterns
        phone_patterns = [
            r'(?:Mob|Phone|Tel|Mobile|Cell)[:\s]*(\+?[\d\s\-().]{10,})',
            r'(\+1\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})',
            r'(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})',
            r'(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
        ]
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phone = re.sub(r'[^\d+\-() ]', '', match.group(1)).strip()
                if len(re.sub(r'\D', '', phone)) >= 10:
                    contact["phone"] = phone
                    break
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/([\w-]+)', text, re.IGNORECASE)
        if linkedin_match:
            contact["linkedin"] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # Location
        loc_patterns = [
            r'([\w\s]+),\s*(TX|CA|NY|PA|IL|OH|GA|NC|FL|WA|MA|India|USA|Karnataka|Maharashtra)',
        ]
        for pattern in loc_patterns:
            match = re.search(pattern, text)
            if match:
                contact["location"] = f"{match.group(1).strip()}, {match.group(2)}"
                break
        
        return contact
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract professional title."""
        # From summary
        summary_match = re.search(
            r'(?:PROFESSIONAL\s+)?SUMMARY[:\s]*\n(.+?)(?:\n[A-Z]{2,}|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(1)
            title_match = re.match(
                r'^([\w\s/]+(?:Manager|Engineer|Developer|Analyst|Consultant|Architect|Lead|Director))\s+with\s+\d+',
                summary.strip(), re.IGNORECASE
            )
            if title_match:
                return title_match.group(1).strip()
        
        return None
    
    def _extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience using multiple strategies."""
        experiences = []
        
        # Strategy 1: "Worked as X in Y from A to B"
        worked_pattern = r'[Ww]ork(?:ed|ing)\s+(?:as\s+)?(?:a\s+)?(.+?)\s+in\s+(.+?)\s+from\s+(\w+\s+\d{4})\s+to\s+(\w+\s+\d{4}|Present|Current)'
        for match in re.finditer(worked_pattern, text, re.IGNORECASE):
            exp = self._build_experience(
                title=match.group(1).strip(),
                employer=match.group(2).strip(),
                start_str=match.group(3),
                end_str=match.group(4)
            )
            if exp:
                experiences.append(exp)
        
        # Strategy 2: Date range detection
        date_pattern = r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*[-–]\s*((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|Present|Current|Till\s+Date)'
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            date_match = re.search(date_pattern, line, re.IGNORECASE)
            if date_match:
                employer, title, location = self._parse_experience_header(line, lines, i, date_match)
                responsibilities = self._extract_responsibilities(lines, i + 1)
                
                exp = self._build_experience(
                    title=title,
                    employer=employer,
                    location=location,
                    start_str=date_match.group(1),
                    end_str=date_match.group(2),
                    responsibilities=responsibilities
                )
                if exp and not self._is_duplicate(exp, experiences):
                    experiences.append(exp)
        
        # Strategy 3: Short date format (Jul24, Jun21)
        short_date_pattern = r'(\w{3}\d{2})\s+to\s+(\w{3}\d{2}|Present|P)'
        for i, line in enumerate(lines):
            if 'Client:' in line:
                match = re.search(short_date_pattern, line, re.IGNORECASE)
                if match:
                    employer_match = re.search(r'Client:\s*(.+?)[-–]', line)
                    if employer_match:
                        exp = self._build_experience(
                            employer=employer_match.group(1).strip(),
                            start_str=match.group(1),
                            end_str=match.group(2),
                            responsibilities=self._extract_responsibilities(lines, i + 1)
                        )
                        if exp and not self._is_duplicate(exp, experiences):
                            experiences.append(exp)
        
        # Sort by start date descending
        experiences.sort(key=lambda x: x.get('start_date', ''), reverse=True)
        return experiences
    
    def _parse_experience_header(self, line: str, lines: List[str], idx: int, date_match) -> Tuple[str, str, str]:
        """Parse experience header line."""
        employer, title, location = "", "", ""
        header = line[:date_match.start()].strip()
        
        # Pipe format: "Title | Date" with company on previous line
        if '|' in line:
            title = line.split('|')[0].strip()
            if idx > 0:
                prev = lines[idx - 1].strip()
                loc_match = re.search(r'[-–]\s*(.+)$', prev)
                if loc_match:
                    employer = prev[:prev.find('-')].strip()
                    location = loc_match.group(1).strip()
                else:
                    employer = prev
        else:
            # Standard: "Company – Title Location"
            dash_match = re.match(r'^([A-Za-z][\w\s&.,()]+?)\s*[-–]\s*(.+)$', header)
            if dash_match:
                employer = dash_match.group(1).strip()
                title = dash_match.group(2).strip()
            else:
                employer = header
        
        return employer, title, location
    
    def _extract_responsibilities(self, lines: List[str], start_idx: int) -> List[str]:
        """Extract bullet points as responsibilities."""
        responsibilities = []
        date_pattern = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\s*[-–]'
        
        for j in range(start_idx, min(start_idx + 25, len(lines))):
            line = lines[j].strip()
            
            if re.search(date_pattern, line, re.IGNORECASE):
                break
            if re.match(r'^(EDUCATION|TECHNICAL|SKILLS|CERTIFICATIONS)', line, re.IGNORECASE):
                break
            
            if line.startswith(('•', '-', '*', '–')) or re.match(r'^\d+\.', line):
                resp = re.sub(r'^[•\-\*–\d.]\s*', '', line)
                if len(resp) > 20:
                    responsibilities.append(resp)
        
        return responsibilities[:12]
    
    def _build_experience(self, title: str = "", employer: str = "", location: str = "",
                         start_str: str = "", end_str: str = "", 
                         responsibilities: List[str] = None) -> Optional[Dict]:
        """Build experience dictionary with calculated duration."""
        if not employer and not title:
            return None
        
        start_year, start_month = self._parse_date(start_str)
        end_year, end_month, is_current = self._parse_date_with_current(end_str)
        
        if not start_year:
            return None
        
        duration = self._calculate_duration(start_year, start_month or 1, end_year, end_month or 12)
        
        return {
            "employer": employer or "Unknown",
            "title": title or "Professional",
            "location": location,
            "start_date": f"{start_year}-{(start_month or 1):02d}",
            "end_date": "Present" if is_current else f"{end_year}-{(end_month or 12):02d}",
            "is_current": is_current,
            "duration_months": duration,
            "responsibilities": responsibilities or [],
            "technologies": []
        }
    
    def _parse_date(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse date string to (year, month)."""
        if not text:
            return None, None
        
        text = text.strip().lower()
        
        # Month Year
        match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s*(\d{4})', text)
        if match:
            return int(match.group(2)), MONTH_MAP.get(match.group(1)[:3])
        
        # Short format: Jul24
        match = re.match(r'(\w{3})(\d{2})$', text)
        if match:
            month = MONTH_MAP.get(match.group(1)[:3].lower())
            year = 2000 + int(match.group(2)) if int(match.group(2)) < 50 else 1900 + int(match.group(2))
            return year, month
        
        # Just year
        match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if match:
            return int(match.group(1)), None
        
        return None, None
    
    def _parse_date_with_current(self, text: str) -> Tuple[Optional[int], Optional[int], bool]:
        """Parse date with current/present detection."""
        if not text:
            return None, None, False
        
        text_lower = text.lower().strip()
        if any(p in text_lower for p in ['present', 'current', 'now', 'till date', 'p']):
            now = datetime.now()
            return now.year, now.month, True
        
        year, month = self._parse_date(text)
        return year, month, False
    
    def _calculate_duration(self, start_year: int, start_month: int, 
                           end_year: int, end_month: int) -> int:
        """Calculate duration in months."""
        if not end_year:
            end_year = datetime.now().year
            end_month = datetime.now().month
        
        start_dt = datetime(start_year, start_month, 1)
        end_dt = datetime(end_year, end_month, 1)
        delta = relativedelta(end_dt, start_dt)
        return max(1, delta.years * 12 + delta.months + 1)
    
    def _is_duplicate(self, exp: Dict, existing: List[Dict]) -> bool:
        """Check if experience is duplicate."""
        for e in existing:
            if e.get('employer') == exp.get('employer') and e.get('start_date') == exp.get('start_date'):
                return True
        return False
    
    def _extract_education(self, text: str) -> List[Dict]:
        """Extract education information."""
        education = []
        
        edu_match = re.search(
            r'(?:EDUCATION(?:AL)?|ACADEMIC)[:\s]*\n?(.+?)(?:\nPROFESSIONAL|\nWORK|\nTECHNICAL|\nSKILLS|\nEXPERIENCE|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if edu_match:
            section = edu_match.group(1)
            
            # Pattern: "Degree | Institution | Year"
            for match in re.finditer(r'([^|\n]+)\s*\|\s*([^|\n]+)\s*\|\s*.*?(\d{4})', section):
                education.append({
                    "degree": match.group(1).strip(),
                    "institution": match.group(2).strip(),
                    "year": match.group(3),
                    "field": None
                })
            
            # Pattern: "Degree – Institution | Year"
            for match in re.finditer(r'([^–\n]+)\s*–\s*([^|\n]+)\s*\|\s*.*?(\d{4})', section):
                if not any(e['degree'] == match.group(1).strip() for e in education):
                    education.append({
                        "degree": match.group(1).strip(),
                        "institution": match.group(2).strip(),
                        "year": match.group(3),
                        "field": None
                    })
            
            # Pattern: "Degree from Institution Year"
            for match in re.finditer(r'((?:Master|Bachelor|MBA|MCA|B\.?Tech|M\.?Tech)[^,\n]+?)\s+(?:from|at)\s+([^,\n]+?)\s+(\d{4})', section, re.IGNORECASE):
                if not any(e['degree'] == match.group(1).strip() for e in education):
                    education.append({
                        "degree": match.group(1).strip(),
                        "institution": match.group(2).strip(),
                        "year": match.group(3),
                        "field": None
                    })
        
        return education
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications."""
        certs = []
        
        cert_match = re.search(
            r'CERTIFICATIONS?[:\s]*\n(.+?)(?:\nPROFESSIONAL|\nEXPERIENCE|\nEDUCATION|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        if cert_match:
            section = cert_match.group(1)
            for line in section.split('\n'):
                line = re.sub(r'^[•·\-\*]\s*', '', line.strip())
                if line and 3 < len(line) < 200:
                    if not re.match(r'^(PROFESSIONAL|EXPERIENCE|IBM|Cognizant)', line, re.IGNORECASE):
                        certs.append(line)
        
        return certs[:15]
    
    def _extract_skills(self, text: str) -> Dict:
        """Extract technical skills."""
        skills = {"technical": [], "tools": [], "domains": []}
        
        # Common skill keywords
        tech_skills = ['python', 'java', 'javascript', 'sql', 'c++', 'c#', 'go', 'rust', 'scala', 'kotlin']
        tools = ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'jira', 'terraform', 'ansible']
        domains = ['banking', 'finance', 'healthcare', 'retail', 'telecom', 'insurance']
        
        text_lower = text.lower()
        
        for skill in tech_skills:
            if re.search(rf'\b{skill}\b', text_lower):
                skills["technical"].append(skill.title() if len(skill) > 3 else skill.upper())
        
        for tool in tools:
            if re.search(rf'\b{tool}\b', text_lower):
                skills["tools"].append(tool.upper() if len(tool) <= 3 else tool.title())
        
        for domain in domains:
            if re.search(rf'\b{domain}\b', text_lower):
                skills["domains"].append(domain.title())
        
        return skills


# ============================================================================
# AGENT 3: VALIDATION AGENT
# ============================================================================

class ValidationAgent:
    """Validates and scores parsed resume data."""
    
    def validate(self, parsed: Dict, original_text: str) -> Tuple[int, List[str], Dict]:
        """Validate parsed data and return score, issues, and fixes."""
        score = 100
        issues = []
        fixes = {}
        
        # Check name
        name = parsed.get("name", {})
        if not name.get("full") or len(name.get("full", "")) < 3:
            score -= 20
            issues.append("missing_name")
        
        # Check contact
        contact = parsed.get("contact", {})
        if not contact.get("email"):
            score -= 15
            issues.append("missing_email")
            # Try to fix
            email_match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', original_text)
            if email_match:
                fixes["email"] = email_match.group(1).lower()
        
        if not contact.get("phone"):
            score -= 10
            issues.append("missing_phone")
        
        # Check experience
        experience = parsed.get("experience", [])
        if len(experience) == 0:
            score -= 25
            issues.append("missing_experience")
        else:
            total_resp = sum(len(e.get("responsibilities", [])) for e in experience)
            if total_resp < 5:
                score -= 10
                issues.append("low_responsibilities")
            
            for i, exp in enumerate(experience):
                if not exp.get("title"):
                    score -= 5
                    issues.append(f"exp_{i}_missing_title")
                if not exp.get("employer"):
                    score -= 5
                    issues.append(f"exp_{i}_missing_employer")
        
        # Check education
        if len(parsed.get("education", [])) == 0:
            score -= 10
            issues.append("missing_education")
        
        # Check title
        if not parsed.get("title"):
            score -= 5
            issues.append("missing_title")
            # Try to derive from experience
            if experience and experience[0].get("title"):
                fixes["title"] = experience[0]["title"]
        
        return max(0, score), issues, fixes


# ============================================================================
# AGENT 4: ENHANCEMENT AGENT
# ============================================================================

class EnhancementAgent:
    """Enhances and normalizes parsed data."""
    
    def enhance(self, parsed: Dict, fixes: Dict) -> Dict:
        """Apply fixes and enhancements."""
        # Apply validation fixes
        if fixes.get("email"):
            if "contact" not in parsed:
                parsed["contact"] = {}
            parsed["contact"]["email"] = fixes["email"]
        
        if fixes.get("title"):
            parsed["title"] = fixes["title"]
        
        # Calculate totals
        experience = parsed.get("experience", [])
        total_months = sum(e.get("duration_months", 0) for e in experience)
        parsed["total_experience_months"] = total_months
        parsed["total_experience_years"] = round(total_months / 12, 1)
        
        # Build full name if missing
        name = parsed.get("name", {})
        if not name.get("full") and (name.get("first") or name.get("last")):
            parts = [name.get("first"), name.get("middle"), name.get("last")]
            name["full"] = ' '.join(filter(None, parts))
            parsed["name"] = name
        
        return parsed


# ============================================================================
# MASTER ORCHESTRATOR
# ============================================================================

class ResumeParserOrchestrator:
    """Orchestrates all agents for complete resume parsing."""
    
    def __init__(self, api_key: str = ""):
        self.extraction_agent = ExtractionAgent(api_key) if api_key else None
        self.rules_agent = RulesAgent()
        self.validation_agent = ValidationAgent()
        self.enhancement_agent = EnhancementAgent()
        self.api_key = api_key
    
    async def parse(self, text: str, filename: str = "") -> Dict:
        """Parse resume using multi-agent system."""
        
        # Step 1: Try AI extraction first (if available)
        ai_result = {}
        if self.extraction_agent:
            ai_result = await self.extraction_agent.extract(text)
        
        # Step 2: Rules-based extraction (always run as backup)
        rules_result = self.rules_agent.extract(text)
        
        # Step 3: Merge results (AI takes priority where available)
        merged = self._merge_results(ai_result, rules_result)
        
        # Step 4: Validate
        score, issues, fixes = self.validation_agent.validate(merged, text)
        
        # Step 5: Enhance
        enhanced = self.enhancement_agent.enhance(merged, fixes)
        
        # Build final output
        return {
            "parsed_resume": self._format_output(enhanced, filename),
            "validation_score": score,
            "validation_issues": issues,
            "extraction_method": "ai+rules" if ai_result else "rules_only"
        }
    
    def _merge_results(self, ai: Dict, rules: Dict) -> Dict:
        """Merge AI and rules results, preferring AI where valid."""
        if not ai:
            return rules
        
        merged = {}
        
        # Name: prefer AI if complete
        ai_name = ai.get("name", {})
        rules_name = rules.get("name", {})
        if ai_name.get("full") and len(ai_name.get("full", "")) > 3:
            merged["name"] = ai_name
        else:
            merged["name"] = rules_name
        
        # Contact: merge both
        merged["contact"] = {**rules.get("contact", {}), **ai.get("contact", {})}
        
        # Title: prefer AI
        merged["title"] = ai.get("title") or rules.get("title")
        
        # Experience: prefer AI if more complete
        ai_exp = ai.get("experience", [])
        rules_exp = rules.get("experience", [])
        
        ai_resp_count = sum(len(e.get("responsibilities", [])) for e in ai_exp)
        rules_resp_count = sum(len(e.get("responsibilities", [])) for e in rules_exp)
        
        if len(ai_exp) >= len(rules_exp) and ai_resp_count >= rules_resp_count:
            merged["experience"] = ai_exp
        else:
            merged["experience"] = rules_exp
        
        # Education: prefer AI if present
        merged["education"] = ai.get("education") or rules.get("education", [])
        
        # Certifications
        merged["certifications"] = ai.get("certifications") or rules.get("certifications", [])
        
        # Skills
        merged["skills"] = ai.get("skills") or rules.get("skills", {})
        
        return merged
    
    def _format_output(self, data: Dict, filename: str) -> Dict:
        """Format output to standard structure."""
        name = data.get("name", {})
        contact = data.get("contact", {})
        
        return {
            "firstname": name.get("first"),
            "lastname": name.get("last"),
            "name": name.get("full"),
            "title": data.get("title"),
            "location": contact.get("location"),
            "phone_number": contact.get("phone"),
            "email": contact.get("email"),
            "linkedin": contact.get("linkedin"),
            "summary": data.get("summary"),
            "total_experience_months": data.get("total_experience_months", 0),
            "total_experience_years": data.get("total_experience_years", 0),
            "experience": [
                {
                    "Employer": e.get("employer"),
                    "title": e.get("title"),
                    "location": e.get("location"),
                    "start_date": e.get("start_date"),
                    "end_date": e.get("end_date"),
                    "duration_months": e.get("duration_months", 0),
                    "responsibilities": e.get("responsibilities", []),
                    "tools": e.get("technologies", [])
                }
                for e in data.get("experience", [])
            ],
            "education": data.get("education", []),
            "certifications": data.get("certifications", []),
            "technical_skills": (
                data.get("skills", {}).get("technical", []) +
                data.get("skills", {}).get("tools", [])
            ),
            "filename": filename
        }


# ============================================================================
# PUBLIC API
# ============================================================================

async def parse_resume_intelligent(text: str, filename: str = "", api_key: str = "") -> Dict:
    """
    Parse resume using intelligent multi-agent system.
    
    Args:
        text: Resume text content
        filename: Original filename
        api_key: Anthropic API key for AI extraction
    
    Returns:
        Parsed resume dictionary
    """
    orchestrator = ResumeParserOrchestrator(api_key)
    return await orchestrator.parse(text, filename)


# For backward compatibility
async def parse_resume_full(params) -> str:
    """Backward compatible wrapper."""
    result = await parse_resume_intelligent(
        text=params.resume_text,
        filename=params.filename or "",
        api_key=""
    )
    return json.dumps(result, indent=2, ensure_ascii=False)
