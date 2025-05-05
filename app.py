import streamlit as st
import base64
from PIL import Image
import io
import random
import pickle
import json
from nlp_utils import extract_resume_info, preprocess_text
from job_utils import calculate_match_scores

st.set_page_config(
    page_title="Hiremeow",
    page_icon="🐱",
    layout="wide",
)

# Load models
@st.cache_resource
def load_models():
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
    return (rf_classifier_categorization, tfidf_vectorizer_categorization)

rf_classifier_categorization, tfidf_vectorizer_categorization = load_models()

def predict_category(resume_text):
    resume_text = preprocess_text(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

st.markdown("""
<style>
    /* Theme variables */
    :root {
        --light-pink: #FFBEE3;      /* Background */
        --dark-pink: #FF85C6;      /* Foreground highlight */
        --black-grey: #333333;      /* Text and key elements */
        --light-grey: #E5E5E5;
    }

    /* Body styling */
    body {
        color: var(--black-grey);
        background-color: var(--light-pink);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        color: var(--black-grey);
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 2rem;
    }

    /* Logo container */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* File uploader styling */
    .stFileUploader > div:first-child {
        border-radius: 10px !important;
        border: 2px dashed var(--dark-pink) !important;
        padding: 2rem !important;
        background-color: var(--light-pink);
    }

    .stFileUploader:hover > div:first-child {
        background-color: #FFF5F7 !important;
    }

    /* Results container */
    .results-container {
        background-color: var(--light-pink);
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: var(--black-grey);
    }

    /* Job tile styling */
    .job-tile {
        background-color: var(--light-pink);
        border: 2px solid var(--dark-pink);
        border-radius: 15px;
        padding: 15px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: var(--black-grey);
    }

    .job-tile:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.15);
    }

    .job-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--black-grey);
        margin-bottom: 10px;
    }

    .job-company {
        font-size: 1rem;
        color: var(--black-grey);
    }

    .match-badge {
        background-color: var(--dark-pink);
        color: white;
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 0.8rem;
        float: right;
    }

    /* Category badge */
    .category-badge {
        background-color: var(--dark-pink);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--black-grey);
        font-size: 1rem;
        margin-top: 2rem;
    }

    /* Section headers */
    .section-header {
        background-color: var(--black-grey);
        color: white;
        padding: 8px 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .skill-tag {
        background-color: var(--dark-pink);
        color: #333333;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    }
</style>
""", unsafe_allow_html=True)

def load_logo_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    return f"data:image/png;base64,{encoded}"

def display_skills(skills):
    skills_html = ""
    for skill in skills:
        skills_html += f'<div class="skill-tag">{skill}</div>'
    st.markdown(skills_html, unsafe_allow_html=True)

# Main 
def main():
    logo_data_url = load_logo_base64("hiremeow_logo.png") 

    st.markdown(f"""
    <div style="text-align: center;">
        <img src="{logo_data_url}" width="500">
    </div>
    <h1 class="main-header">
        Hiremeow Resume Analyzer
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; color: #333333;'>Upload your resume to find the predicted category, extracted information and job recommendation</p>", unsafe_allow_html=True)
    
    # center
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("", type=['txt'])
        
    if uploaded_file is not None:
        resume_text = uploaded_file.read().decode("utf-8")
        
        with st.spinner("loading..."):
            predicted_category = predict_category(resume_text)
            resume_info = extract_resume_info(resume_text)
            import time
            time.sleep(1.5)
            
            st.markdown("<div class='results-container'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <h3>Your Resume Category</h3>
                <div class='category-badge'>{predicted_category}</div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<h2 style='margin-top: 2rem; text-align: center;'>Jobs You Can Apply To</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-bottom: 2rem;'>These positions match your skills and experience</p>", unsafe_allow_html=True)

            # load job data from jobs.json
            try:
                with open('jobs.json', 'r') as f:
                    data = json.load(f)
                    if 'job_listings' in data:
                        jobs = data['job_listings']
            except FileNotFoundError:
                st.error("jobs.json file not found!")
                st.stop()
            except json.JSONDecodeError:
                st.error("Error decoding the jobs.json file. Please check the format.")
                st.stop()
            
            # convert dict to list
            if isinstance(jobs, dict):
                jobs_list = [{"id": job_id, **job_data} for job_id, job_data in jobs.items()]
            else:
                jobs_list = jobs
                
            matched_jobs = calculate_match_scores(jobs_list, resume_info['skills'])
                    
            job_cols = st.columns(3)
    
            for i, job in enumerate(matched_jobs):
                col_idx = i % 3
                with job_cols[col_idx]:
                    st.markdown(f"""
                    <div class='job-tile'>
                        <span class='match-badge'>{job.get('match_score', 0)}% Match</span>
                        <div class='job-title'>{job.get('title', 'Unknown Position')}</div>
                        <div class='job-company'>{job.get('company', 'Unknown Company')}</div>
                        <div>{job.get('short_description', '')[:100]}{'...' if len(job.get('short_description', '')) > 100 else ''}</div>
                        <div class='job-skills'>
                            {' '.join(f"<span class='skill-tag'>{skill}</span>" for skill in job.get('required_skills', [])[:])}
                            {f"<span class='skill-tag'>+{len(job.get('required_skills', [])) - 3} more</span>" if len(job.get('required_skills', [])) > 3 else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Resume INfo
            st.markdown("## Resume Analysis")

            # Skills 
            st.markdown("### Skills")
            display_skills(resume_info['skills'])
            
            # Education
            st.markdown("### Education")
            for edu in resume_info['education']:
                st.markdown(f"- **{edu['degree']}** - {edu['institution']}")
                if 'year' in edu:
                    st.markdown(f"  {edu['year']}")
            
            # Experience 
            st.markdown("### Experience")
            for exp in resume_info['experience']:
                st.markdown(f"- **{exp['title']}** at {exp['company']}")
                if 'duration' in exp:
                    st.markdown(f"  {exp['duration']}")

            # Text 
            st.markdown("### Preprocessed Resume Text")
            st.markdown(f"""
                <div><p>{resume_info['text']}</p></div>
                """, unsafe_allow_html=True)
            
                    
    
    # Footer
    st.markdown("""
    <div class='footer'>
        Created by Yeshaswi <3
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()