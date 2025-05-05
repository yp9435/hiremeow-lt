import spacy
import json

# Load model
nlp = spacy.load('en_core_web_sm')
stop_words_spacy = nlp.Defaults.stop_words

def spacy_keywords(data):
    if isinstance(data, list):
        data = ' '.join([item.lower() for item in data])
    else:
        data = data.lower()
    
    tokens = nlp(data)
    pos_tagged_tokens = [(tok, tok.tag_) for tok in tokens]
    keywords = [str(t[0]).lower() for t in pos_tagged_tokens if t[1] in ['NNP', 'NN']]
    keywords = [w for w in keywords if w not in stop_words_spacy]
    keywords = sorted(list(set(keywords)))
    return keywords

def calculate_match_scores(job_listings, extracted_skills):
    if isinstance(job_listings, dict):
        jobs_list = []
        for job_id, job_data in job_listings.items():
            job_data['id'] = job_id
            jobs_list.append(job_data)
    else:
        jobs_list = job_listings

    extracted_skills = [skill.lower() for skill in extracted_skills]

    for job in jobs_list:
        required_skills = [skill.lower() for skill in job.get('required_skills', [])]
        skills_keywords = spacy_keywords(required_skills)
        
        if 'short_description' in job:
            desc_keywords = spacy_keywords(job['short_description'])
            all_job_keywords = list(set(skills_keywords + desc_keywords))
        else:
            all_job_keywords = skills_keywords
        
        matched_keywords = set(all_job_keywords).intersection(set(extracted_skills))
        
        if len(extracted_skills) > 0:
            match_percentage = int((len(matched_keywords) / len(extracted_skills)) * 100)
        else:
            match_percentage = 0
        
        job['match_score'] = min(match_percentage, 100)
        job['matched_keywords'] = list(matched_keywords)

    sorted_jobs = sorted(jobs_list, key=lambda x: x.get('match_score', 0), reverse=True)
    return sorted_jobs
