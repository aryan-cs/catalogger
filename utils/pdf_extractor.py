import requests
import io
import re
from pypdf import PdfReader

def extract_text_from_first_page(pdf_url):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        
        with io.BytesIO(response.content) as f:
            reader = PdfReader(f)
            if len(reader.pages) > 0:
                return reader.pages[0].extract_text()
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_url}: {e}")
        return ""
    return ""

def find_emails_in_text(text):
    emails = []
    
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    found = re.findall(email_regex, text)
    emails.extend(found)
    
    curly_regex = r'\{([^{}]+)\}@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
    curly_matches = re.findall(curly_regex, text)
    for users, domain in curly_matches:
        user_list = [u.strip() for u in users.split(',')]
        for user in user_list:
            emails.append(f"{user}@{domain}")
            
    return list(set(emails))

def get_emails_from_pdf(pdf_url):
    text = extract_text_from_first_page(pdf_url)
    if text:
        return find_emails_in_text(text)
    return []
