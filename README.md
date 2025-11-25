# Catalogger

Catalogger is a Streamlit-based research concierge for NeurIPS/ICLR/ICML papers. It lets you load conference datasets, perform semantic search using sentence-transformer embeddings, and generate AI-curated recommendations enriched with contact info (email, X) pulled from paper PDFs and author discovery workflows.


## Features

- **Conference Loader** – fetches the latest OpenReview data (cached locally under `data/`).
- **Semantic Search** – sentence-transformer embeddings power contextual matching on abstracts + titles.
- **AI Recommendations** – Gemini 2.5 Pro or OpenRouter GPT-4o-mini (not tested) summarizes the top hits with keywords, relevance, and icebreakers.
- **Contact Enrichment** – scans the PDF first page for author emails and surfaces Twitter/X links (or a one-click search fallback).
- **UI Enhancements** – keyword pills to append interest tags, “Generate more” button to stack additional recs, dark theme, and pinned sidebar footer.

## Local Setup

```bash
# clone
 git clone https://github.com/aryan-cs/catalogger.git
 cd catalogger

# create venv
 python -m venv .venv
 .venv\Scripts\activate      # Windows
 
# install deps
 pip install -r requirements.txt

# run streamlit
 streamlit run app.py
```

Optional: pre-fetch a conference once to warm the cache (`Load Conference Data` button in the sidebar).

## Environment / API Keys

The app supports two providers:

- **Google Gemini** – set `GEMINI_API_KEY` in Streamlit secrets or enter via sidebar.
- **OpenRouter** – set `OPENROUTER_API_KEY` similarly.

When deploying on Streamlit Cloud, define these in *Settings → Secrets* as:

```ini
GEMINI_API_KEY = "..."
OPENROUTER_API_KEY = "..."
```

In development, you can also provide the key via the sidebar input (stored only in session state).

## Project Structure

```
app.py                  # main Streamlit UI
utils/
  ├── data_fetcher.py   # OpenReview loaders + author profiles
  ├── search_engine.py  # embedding index + semantic search
  ├── llm_interface.py  # Gemini/OpenRouter wrapper
  └── pdf_extractor.py  # downloads PDFs and extracts first-page emails
requirements.txt        # python dependencies
streamlit/              # theme config, secrets placeholder
```

## Credits

This app was made by [@aryan-cs](https://github.com/aryan-cs) and Gemini 3 Pro.

## Contributing

See anything you want to fix? Feel free to [open a pull request](https://github.com/aryan-cs/catalogger/pulls) here.
