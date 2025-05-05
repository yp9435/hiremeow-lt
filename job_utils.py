import spacy
import json

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
stop_words_spacy = nlp.Defaults.stop_words

def spacy_keywords(data):
    if isinstance(data, list):
        # If data is a list (like required_skills), join it into a string
        data = ' '.join(data)
    
    tokens = nlp(data.lower())
    pos_tagged_tokens = [(tok, tok.tag_) for tok in tokens]
    keywords = [str(t[0]) for t in pos_tagged_tokens if t[1] in ['NNP', 'NN']]
    keywords = [w for w in keywords if w not in stop_words_spacy]
    keywords = sorted(list(set(x for x in keywords)))
    return keywords

def calculate_match_scores(job_listings, extracted_skills):
    # Convert job listings dictionary to list if needed
    if isinstance(job_listings, dict):
        jobs_list = []
        for job_id, job_data in job_listings.items():
            job_data['id'] = job_id
            jobs_list.append(job_data)
    else:
        jobs_list = job_listings
    
    # Process each job and calculate match score
    for job in jobs_list:
        # Get keywords from job required skills and description
        skills_keywords = spacy_keywords(job.get('required_skills', []))
        
        # Include description keywords too if available
        if 'short_description' in job:
            desc_keywords = spacy_keywords(job['short_description'])
            all_job_keywords = list(set(skills_keywords + desc_keywords))
        else:
            all_job_keywords = skills_keywords
        
        # Calculate match score
        matched_keywords = set(all_job_keywords).intersection(set(extracted_skills))
        
        # Calculate match percentage (as an integer)
        if len(extracted_skills) > 0:
            match_percentage = int((len(matched_keywords) / len(extracted_skills)) * 100)
        else:
            match_percentage = 0
            
        # Cap match score at 100%
        match_percentage = min(match_percentage, 100)
        
        # Store match score and matched keywords in job
        job['match_score'] = match_percentage
        job['matched_keywords'] = list(matched_keywords)
    
    # Sort jobs by match score in descending order
    sorted_jobs = sorted(jobs_list, key=lambda x: x.get('match_score', 0), reverse=True)
    
    return sorted_jobs
