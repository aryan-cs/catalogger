import openreview
import pandas as pd
import os
from typing import List, Dict
import time

DATA_DIR = "data"

CONFERENCE_MAP = {
    "NeurIPS 2025": "NeurIPS.cc/2025/Conference/-/Submission",
    "NeurIPS 2024": "NeurIPS.cc/2024/Conference/-/Submission",
    "NeurIPS 2023": "NeurIPS.cc/2023/Conference/-/Submission", 
}

def get_local_file_path(conference_name: str) -> str:
    safe_name = conference_name.replace(" ", "_").lower()
    return os.path.join(DATA_DIR, f"{safe_name}.csv")

def get_cached_conferences() -> Dict[str, int]:
    cached = {}
    for conf_name in CONFERENCE_MAP.keys():
        path = get_local_file_path(conf_name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=['id'])
                cached[conf_name] = len(df)
            except:
                pass
    return cached

def load_papers(conference_name: str) -> pd.DataFrame:
    file_path = get_local_file_path(conference_name)
    
    if os.path.exists(file_path):
        print(f"Loading {conference_name} from local cache...")
        return pd.read_csv(file_path)
    
    print(f"Fetching {conference_name} from OpenReview (this may take a while)...")
    return fetch_and_save_papers(conference_name)

def fetch_and_save_papers(conference_name: str) -> pd.DataFrame:
    invitation_id = CONFERENCE_MAP.get(conference_name)
    if not invitation_id:
        raise ValueError(f"Unknown conference: {conference_name}")

    print(f"Connecting to OpenReview (API V2) to fetch papers for {conference_name} (ID: {invitation_id})...")
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    
    notes = client.get_all_notes(invitation=invitation_id)
    
    papers_data = []
    for note in notes:
        title = note.content.get('title', {}).get('value', '')
        abstract = note.content.get('abstract', {}).get('value', '')
        authors = ", ".join(note.content.get('authors', {}).get('value', []))
        author_emails = ", ".join(note.content.get('authorids', {}).get('value', []))
        keywords = ", ".join(note.content.get('keywords', {}).get('value', []))
        
        paper = {
            "id": note.id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "author_emails": author_emails,
            "keywords": keywords,
            "pdf_url": f"https://openreview.net/pdf?id={note.id}"
        }
        papers_data.append(paper)
        
    if not papers_data:
        raise ValueError(f"No papers found for {conference_name} with invitation {invitation_id}. Please check if the conference ID is correct or if the papers are public.")

    df = pd.DataFrame(papers_data)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    file_path = get_local_file_path(conference_name)
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} papers to {file_path}")
    
    return df

def get_author_details(author_ids: List[str]) -> Dict[str, str]:
    if not author_ids:
        return {}
        
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    
    profile_ids = [aid for aid in author_ids if aid.startswith('~')]
    emails_map = {aid: aid for aid in author_ids if '@' in aid}
    
    if not profile_ids:
        return emails_map
        
    try:
        profiles = client.get_profiles(ids=profile_ids)
        
        for profile in profiles:
            public_email = profile.content.get('public_email')
            if public_email:
                emails_map[profile.id] = public_email
            elif profile.content.get('emails'):
                emails_map[profile.id] = profile.content['emails'][0]
            else:
                emails_map[profile.id] = None
                
    except Exception as e:
        print(f"Error fetching profiles: {e}")
        
    return emails_map
