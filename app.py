import streamlit as st
import pandas as pd
import os
import json
from googlesearch import search
from utils.data_fetcher import load_papers, get_cached_conferences, CONFERENCE_MAP, get_author_details
from utils.search_engine import SearchEngine
from utils.llm_interface import LLMInterface
from utils.pdf_extractor import get_emails_from_pdf

def deduplicate_recommendations(recommendations):
    seen = set()
    unique = []
    for rec in recommendations:
        key = rec.get('id') or rec.get('title')
        if not key:
            unique.append(rec)
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique

st.set_page_config(page_title="Catalogger", layout="wide")

if 'loaded_conferences' not in st.session_state:
    st.session_state['loaded_conferences'] = get_cached_conferences()

for key in list(st.session_state.keys()):
    if key.startswith("kw_") and st.session_state[key]:
        selected = st.session_state[key]
        current_text = st.session_state.get('interests_input', "")
        if selected not in current_text:
            if current_text:
                st.session_state['interests_input'] = current_text + ", " + selected
            else:
                st.session_state['interests_input'] = selected
        st.session_state[key] = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Stack+Sans+Text:wght@300;400;500;600;700&display=swap');

html, body, h1, h2, h3, h4, h5, h6, p, label, button, input, textarea {
    font-family: 'Stack Sans Text', sans-serif !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #4a4a4a !important;
    border-color: #4a4a4a !important;
}

div[data-testid="stMarkdown"] h3 a {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Catalogger")
st.markdown("Prepare for your next conference by searching through thousands of papers and find what interests you.")
st.divider()

with st.sidebar:
    st.header("Configuration")
    
    api_provider = st.selectbox("AI Provider", ["Google Gemini", "OpenRouter"])
    api_key = st.text_input("API Key", type="password")
    
    st.divider()
    
    conference = st.selectbox(
        "Select Conference", 
        list(CONFERENCE_MAP.keys())
    )
    
    if st.button("Load Conference Data", use_container_width=True):
        with st.spinner(f"Loading data for {conference}..."):
            try:
                df = load_papers(conference)
                st.session_state['df'] = df
                st.session_state['conference'] = conference
                
                if 'loaded_conferences' not in st.session_state:
                    st.session_state['loaded_conferences'] = {}
                st.session_state['loaded_conferences'][conference] = len(df)
                
                st.session_state['success_msg'] = f"Loaded {len(df)} papers!"
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")

    if 'success_msg' in st.session_state:
        st.success(st.session_state['success_msg'])
        del st.session_state['success_msg']

    st.divider()
    st.write("Loaded Conferences")
    
    if 'loaded_conferences' in st.session_state and st.session_state['loaded_conferences']:
        sorted_confs = sorted(st.session_state['loaded_conferences'].items(), key=lambda x: x[0], reverse=True)
        
        for conf_name, count in sorted_confs:
            is_active = ('conference' in st.session_state and st.session_state['conference'] == conf_name)
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(conf_name, key=f"load_{conf_name}", type=btn_type, use_container_width=True):
                with st.spinner(f"Switching to {conf_name}..."):
                    try:
                        df = load_papers(conf_name)
                        st.session_state['df'] = df
                        st.session_state['conference'] = conf_name
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading data: {e}")
    else:
        st.caption("No papers loaded.")
    
    st.divider()
    st.caption("Vibe coded by [@aryanguptacs](https://x.com/aryanguptacs) using Gemini 3 Pro.")
    
if 'df' in st.session_state:
    df = st.session_state['df']
    
    if 'search_engine' not in st.session_state or st.session_state.get('se_conf') != st.session_state['conference']:
        se = SearchEngine()
        
        safe_name = st.session_state['conference'].replace(" ", "_").lower()
        emb_path = os.path.join("data", f"{safe_name}_embeddings.pkl")
        needs_indexing = not os.path.exists(emb_path)
        
        if needs_indexing:
            progress_bar = st.progress(0, text="Initializing semantic search engine...")
            
            def update_progress(progress, text):
                progress_bar.progress(progress, text=text)
                
            se.load_data(df, st.session_state['conference'], progress_callback=update_progress)
            progress_bar.empty()
        else:
            with st.spinner("Loading cached search index..."):
                se.load_data(df, st.session_state['conference'])
            
        st.session_state['search_engine'] = se
        st.session_state['se_conf'] = st.session_state['conference']
    
    if 'interests_input' not in st.session_state:
        st.session_state['interests_input'] = ""
    if 'candidates' not in st.session_state:
        st.session_state['candidates'] = None
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = None
    if 'append_mode' not in st.session_state:
        st.session_state['append_mode'] = False

    interests = st.text_area(
        f"What topics from {st.session_state['conference']} are you interested in?", 
        placeholder="e.g. I'm interested in reinforcement learning for robotics, specifically sim-to-real transfer and transformer architectures.",
        height=100,
        key="interests_input"
    )
    
    def run_search():
        if not interests:
            st.error("Please enter your interests.")
            return
        
        se = st.session_state['search_engine']
        candidates = se.search(interests, top_k=50)
        st.session_state['candidates'] = candidates
        
        st.session_state['recommendations'] = None
        st.session_state['run_llm'] = True
        st.session_state['append_mode'] = False

    st.button("Find Matches", type="primary", use_container_width=True, on_click=run_search)

    if st.session_state['candidates'] is not None:
        candidates = st.session_state['candidates']
        
        with st.expander("See raw candidate papers (Top 20)", expanded=True):
            display_df = candidates[['title', 'authors', 'keywords', 'similarity_score', 'pdf_url']].head(20).copy()
            display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.1%}")
            st.dataframe(
                display_df,
                column_config={
                    "title": "Title",
                    "authors": "Authors",
                    "keywords": "Keywords",
                    "similarity_score": "Similarity Score",
                    "pdf_url": st.column_config.LinkColumn("Link", display_text="View Paper"),
                }
            )
        
        st.divider()
        
        if st.session_state.get('run_llm'):
            append_mode = st.session_state.get('append_mode', False)
            previous_recs = st.session_state['recommendations'] if append_mode and isinstance(st.session_state.get('recommendations'), list) else None
            if api_key:
                model_display = "Gemini 2.5 Pro" if api_provider == "Google Gemini" else "GPT-4o Mini via OpenRouter"
                with st.spinner(f"Generating customized recommendations using {model_display}..."):
                    llm = LLMInterface(api_provider, api_key)
                    analysis_json = llm.analyze_papers(candidates, interests)
                    
                    if analysis_json.startswith("Error"):
                        st.session_state['recommendations'] = {"error": analysis_json}
                    else:
                        try:
                            cleaned_json = analysis_json.strip()
                            if cleaned_json.startswith("```json"):
                                cleaned_json = cleaned_json[7:]
                            elif cleaned_json.startswith("```"):
                                cleaned_json = cleaned_json[3:]
                            if cleaned_json.endswith("```"):
                                cleaned_json = cleaned_json[:-3]
                            
                            recommendations = json.loads(cleaned_json)
                            
                            progress_text = "Fetching author emails from PDFs..."
                            enrich_bar = st.progress(0, text=progress_text)
                            
                            for p_idx, paper in enumerate(recommendations):
                                enrich_bar.progress((p_idx) / len(recommendations), text=f"Fetching info for paper {p_idx+1}/{len(recommendations)}")
                                paper_id = paper.get('id')
                                if paper_id is not None:
                                    try:
                                        pid_int = int(paper_id)
                                        if pid_int in candidates.index:
                                            row = candidates.loc[pid_int]
                                            paper_url = row.get('pdf_url', '#')
                                            paper['url'] = paper_url
                                            
                                            if paper_url and paper_url != '#':
                                                found_emails = get_emails_from_pdf(paper_url)
                                                
                                                for auth_obj in paper['authors']:
                                                    auth_name = auth_obj['name']
                                                    
                                                    try:
                                                        name_parts = auth_name.lower().split()
                                                        if name_parts:
                                                            last_name = name_parts[-1]
                                                            last_name = "".join(c for c in last_name if c.isalnum())
                                                            
                                                            for email in found_emails:
                                                                if last_name in email.lower():
                                                                    auth_obj['email'] = email
                                                                    break
                                                    except:
                                                        pass

                                                    try:
                                                        query = f"{auth_name} x.com"
                                                        results = list(search(query, num_results=5))
                                                        print(f"Twitter search for {auth_name}: {len(results)} results")
                                                        found_handle = False
                                                        for result in results:
                                                            normalized = result.split('?')[0].rstrip('/')
                                                            if "twitter.com" in normalized or "x.com" in normalized:
                                                                parts = normalized.split('/')
                                                                handle = parts[-1]
                                                                if handle and handle.lower() not in ("home", "i", "login"):
                                                                    auth_obj['twitter'] = handle
                                                                    auth_obj['twitter_url'] = normalized
                                                                    found_handle = True
                                                                    break
                                                        if not found_handle:
                                                            print(f"No twitter found for {auth_name}. Results: {results}")
                                                    except Exception as tw_err:
                                                        print(f"Twitter search failed for {auth_name}: {tw_err}")

                                    except Exception as e:
                                        print(f"Error enriching paper {paper_id}: {e}")
                            
                            enrich_bar.empty()
                            if previous_recs:
                                combined = previous_recs + recommendations
                            else:
                                combined = recommendations
                            st.session_state['recommendations'] = deduplicate_recommendations(combined)
                            
                        except json.JSONDecodeError:
                            st.session_state['recommendations'] = {"error": "Failed to parse AI response.", "raw": analysis_json}
                        except Exception as e:
                            st.session_state['recommendations'] = {"error": f"An error occurred: {e}", "raw": analysis_json}
            else:
                st.session_state['recommendations'] = "NO_API_KEY"
            
            st.session_state['run_llm'] = False
            st.session_state['append_mode'] = False

        
        recs = st.session_state['recommendations']
        
        if recs == "NO_API_KEY":
            st.warning("Please enter an API Key in the sidebar to get customized AI recommendations and author contact info.")
        elif isinstance(recs, dict) and "error" in recs:
            st.error(recs["error"])
            if "raw" in recs:
                st.markdown(recs["raw"])
        elif isinstance(recs, list):
            for i, paper in enumerate(recs):
                with st.container(border=True):
                    paper_url = paper.get('url', '#')
                    st.markdown(f"### [{paper['title']}]({paper_url})")
                    
                    authors_formatted = []
                    for a in paper['authors']:
                        name = a['name']
                        email = a.get('email')
                        twitter = a.get('twitter')
                        twitter_url = a.get('twitter_url')
                        
                        extras = []
                        if email:
                            extras.append(email)
                        if twitter and twitter_url:
                            extras.append(f"[@{twitter}]({twitter_url})")
                        else:
                            fallback_query = name.replace(' ', '+') + "+researcher+x.com"
                            fallback_url = f"https://www.google.com/search?q={fallback_query}"
                            extras.append(f"[X]({fallback_url})")
                        
                        authors_formatted.append(f"**{name}** ({', '.join(extras)})")
                    st.markdown(", ".join(authors_formatted))
                    
                    if 'keywords' in paper and paper['keywords']:
                        pill_key = f"kw_{i}_{paper['title'][:10]}"
                        st.pills("Keywords", paper['keywords'], selection_mode="single", key=pill_key)
                    
                    st.markdown(f"**Why it's relevant:** {paper['relevance']}")
                    st.info(f"**Icebreaker:** {paper['icebreaker']}")
            
            if api_key:
                if st.button("Generate more recommendations", use_container_width=True):
                    st.session_state['append_mode'] = True
                    st.session_state['run_llm'] = True
                    st.rerun()
                
else:
    st.info("To begin, select a conference to search from.")
