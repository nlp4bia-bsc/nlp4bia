import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import random
import torch

class FaissIndex:
    def __init__(self, model: SentenceTransformer, gazetteer_path: str, use_gpu: bool = False, random_seed: int = 42,
                training_data_path:str=None):
        self.model_st = model
        self.gazetteer_df = pd.read_csv(gazetteer_path, sep="\t")
        if training_data_path: 
            self.train_df = pd.read_csv(training_data_path, sep="\t")
            self._add_training_to_gazetteer()
        self.use_gpu = use_gpu
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        self._set_random_seeds(self.random_seed)
                
        # Maintain a mapping of index to codes
        self.index_id_to_code = {i: str(self.gazetteer_df.iloc[i]['code']) for i in range(len(self.gazetteer_df))}

    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _add_training_to_gazetteer(self):
        # Filter entries with source=="SCTID"
        filtered_df = self.train_df[self.train_df['source'] == 'SCTID']
        # Create df without duplicate entries
        train_df_to_add = filtered_df[['code', 'span']].drop_duplicates().copy()
        # Add required columns:
        # Añadir las columnas requeridas
        train_df_to_add['semantic_tag'] = "from_training"
        train_df_to_add['mainterm'] = 0
        train_df_to_add['language'] = 'es'
        train_df_to_add = train_df_to_add[['code', 'span', 'semantic_tag', 'mainterm', 'language']]
        train_df_to_add.columns = ['code', 'term', 'semantic_tag', 'mainterm', 'language']           

        # Add that dataframe to gazeteer.
        self.gazetteer_df = pd.concat([self.gazetteer_df,train_df_to_add]).drop_duplicates(subset=["term","code"]).copy().reset_index(drop=True)


    def generate_search_index(self):
        # Generate embeddings for the gazetteer terms
        terms_gaz  = self.gazetteer_df['term'].to_list()
        embeddings = self.model_st.encode(terms_gaz, show_progress_bar=True)
        self.gazetteer_df['embedding'] = embeddings.tolist()
        
        # Initialize FAISS index
        self.dimension = len(embeddings[0])
        self.index_faiss = faiss.IndexFlatIP(self.dimension)  # Inner Product for cosine similarity
        
        # Move the index to the GPU if necessary
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # Initialize GPU resources
            self.index_faiss = faiss.index_cpu_to_gpu(res, 0, self.index_faiss)  # 0 is the GPU ID
        
        # Add vectors to the index
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index_faiss.add(embeddings)
    
    def search(self, term_embeddings: np.ndarray, k: int = 5):
        # Perform the search on the index
        faiss.normalize_L2(term_embeddings)
        D, I = self.index_faiss.search(term_embeddings, k)
        return D, I
    
    def get_code_by_index(self, index: int) -> str:
        # Retrieve the code associated with a specific index
        return self.index_id_to_code[index]
    
    def get_term_by_index(self, index: int) -> str:
        # Retrieve the term associated with a specific index
        return self.gazetteer_df.iloc[index]['term']
    
    def write(self, file_path: str):
        # Ensure the index is on CPU before saving
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index_faiss)
        else:
            index_cpu = self.index_faiss
        
        # Create the directory if it does not exist
        file_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        faiss.write_index(index_cpu, file_path)
        print(f"Index saved to {file_path}")

    def read(self, file_path: str):
        # Read the index from disk
        index_cpu = faiss.read_index(file_path)
        
        # Move the index to GPU if necessary
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # Initialize GPU resources
            self.index_faiss = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # 0 is the GPU ID
        else:
            self.index_faiss = index_cpu
        
        print(f"Index loaded from {file_path}")
