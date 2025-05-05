import re
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

#loading spacy and tranformer models
nlp = spacy.load("en_core_web_lg")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text, skill_patterns=None):
    
    # list of predefined skills
    skills_list = [
        "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", "kotlin",
        "react", "angular", "vue", "django", "flask", "spring", "node.js", "express",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
        "sql", "mysql", "postgresql", "mongodb", "oracle", "firebase", "aws", "azure",
        "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab", "jira",
        "agile", "scrum", "kanban", "html", "css", "sass", "less", "bootstrap",
        "jquery", "typescript", "redux", "graphql", "rest", "soap", "json", "xml",
        "linux", "unix", "windows", "macos", "android", "ios", "flutter", "react native",
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "data analysis", "data visualization", "big data", "hadoop", "spark", "kafka",
        "tableau", "power bi", "excel", "word", "powerpoint", "photoshop", "illustrator",
        "figma", "sketch", "adobe xd", "ui/ux", "responsive design", "seo", "sem",
        "digital marketing", "content marketing", "social media marketing", "email marketing",
        "project management", "product management", "leadership", "teamwork", "communication",
        "problem solving", "critical thinking", "creativity", "time management", "organization"
    ]
    
    # spacyyy
    doc = nlp(text)
    skills = []
    
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            skills.append(skill)

    #goes through all the enitities in the doc and tries to find skills
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            if ent.text.lower() in skills_list:
                skills.append(ent.text.lower())
    
    skills = sorted(list(set(skills)))
    
    return skills

def extract_education(text):
    education_info = []

    education_section = ""
    blocks = re.split(r"\n{2,}|\r\n{2,}", text)
    for block in blocks:
        if "education" in block.lower():
            education_section = block
            break

    if not education_section:
        education_section = text

    lines = education_section.split("\n")
    degree_keywords = [
        "bachelor", "master", "phd", "mba", "b.tech", "m.tech", "b.sc", "m.sc", "cbse",
        "b tech", "m tech", "b.a", "m.a", "b.com", "m.com"
    ]

    year_pattern = r"(19|20)\d{2}"
    
    for line in lines:
        line_lower = line.lower()
        if any(deg in line_lower for deg in degree_keywords):
            degree_match = re.search(r"(b\.?\s?tech|m\.?\s?tech|bachelor(?:'s)?|master(?:'s)?|ph\.?d\.?|mba|b\.?sc\.?|m\.?sc\.?|cbse|b\.?a\.?|m\.?a\.?|b\.?com|m\.?com)", line_lower)
            year_match = re.findall(year_pattern, line)
            degree = degree_match.group(0).upper() if degree_match else "Unknown Degree"

            institution = "Unknown Institution"
            doc = nlp(line)
            for ent in doc.ents:
                if ent.label_ == "ORG" and any(k in ent.text.lower() for k in ["university", "college", "institute", "school", "academy", "cbse"]):
                    institution = ent.text
                    break
            else:
                institution_guess = " ".join(line.split()[:6])  # Take the first few words
                institution = institution_guess

            education_info.append({
                "degree": degree.title(),
                "institution": institution,
                "year": re.search(year_pattern, line).group(0) if re.search(year_pattern, line) else None
            })

    return education_info

def extract_experience(text):
    doc = nlp(text)
    
    experience_keywords = [
        "experience", "work", "employment", "job", "position", "role", "career",
        "professional", "occupation", "worked", "employed", "intern", "internship"
    ]
    
    job_title_patterns = [
        r"(?i)(software engineer|software engineering intern|developer|programmer|analyst|designer|architect|manager|director|lead|consultant|specialist|administrator|technician|officer|coordinator|assistant|associate|senior|junior|chief|head|vp|vice president|ceo|cto|cio|cfo|president|founder|co-founder)"
    ]
    
    duration_pattern = r"(?i)(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s+(?:to|–|-)\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\s+(?:to|–|-)\s+present|(?:19|20)\d{2}\s+(?:to|–|-)\s+(?:19|20)\d{2}|(?:19|20)\d{2}\s+(?:to|–|-)\s+present|\d+\s+years?|\d+\s+months?"
    
    experience_sections = []
    
    paragraphs = text.split("\n\n")
    
    for paragraph in paragraphs:
        if any(keyword in paragraph.lower() for keyword in experience_keywords):
            experience_sections.append(paragraph)
    
    experience_info = []
    
    for section in experience_sections:
        title = "Job Title"
        for pattern in job_title_patterns:
            match = re.search(pattern, section)
            if match:
                title = match.group(0)
                break
        
        company = "Company"
        for ent in nlp(section).ents:
            if ent.label_ == "ORG":
                # Check if the entity is likely a company
                if not any(keyword in ent.text.lower() for keyword in ["university", "college", "institute", "school"]):
                    company = ent.text
                    break
        
        duration = None
        duration_match = re.search(duration_pattern, section)
        if duration_match:
            duration = duration_match.group(0)
        
        experience_info.append({
            "title": title,
            "company": company,
            "duration": duration
        })
    
    # extract from the entire text
    if not experience_info:
        title = "Job Title"
        for pattern in job_title_patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(0)
                break
        
        company = "Company"
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Check if the entity is likely a company
                if not any(keyword in ent.text.lower() for keyword in ["university", "college", "institute", "school"]):
                    company = ent.text
                    break
        
        duration = None
        duration_match = re.search(duration_pattern, text)
        if duration_match:
            duration = duration_match.group(0)
        
        experience_info.append({
            "title": title,
            "company": company,
            "duration": duration
        })
    
    return experience_info

def extract_resume_info(text):
    preprocessed_text = preprocess_text(text)
    skills = extract_skills(preprocessed_text)
    education = extract_education(text)
    experience = extract_experience(text)
    resume_info = {
        "skills": skills,
        "education": education,
        "experience": experience,
        "text": preprocessed_text
    }
    print(resume_info)
    return resume_info