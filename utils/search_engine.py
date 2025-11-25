import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

class SearchEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embeddings = None
        self.df = None
        self.conference_name = None

    def load_data(self, df: pd.DataFrame, conference_name: str, progress_callback=None):
        self.df = df
        self.conference_name = conference_name
        self._load_or_compute_embeddings(progress_callback)

    def _load_or_compute_embeddings(self, progress_callback=None):
        safe_name = self.conference_name.replace(" ", "_").lower()
        emb_path = os.path.join("data", f"{safe_name}_embeddings.pkl")

        if os.path.exists(emb_path):
            with open(emb_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            if progress_callback:
                progress_callback(1.0, "Loaded cached embeddings!")
        else:
            print("Computing embeddings (this happens once per conference)...")
            titles = self.df['title'].fillna('')
            abstracts = self.df['abstract'].fillna('')
            text_data = (titles + ". " + abstracts).tolist()
            
            batch_size = 32
            total_papers = len(text_data)
            embeddings_list = []
            
            for i in range(0, total_papers, batch_size):
                batch = text_data[i : i + batch_size]
                batch_emb = self.model.encode(batch)
                embeddings_list.append(batch_emb)
                
                if progress_callback:
                    progress = min((i + batch_size) / total_papers, 1.0)
                    progress_callback(progress, f"Indexing paper {min(i + batch_size, total_papers)}/{total_papers}...")
            
            self.embeddings = np.vstack(embeddings_list)
            
            with open(emb_path, 'wb') as f:
                pickle.dump(self.embeddings, f)

    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        if self.embeddings is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        
        return results
