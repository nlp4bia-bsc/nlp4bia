import torch
import numpy as np

class DenseRetriever:
    def __init__(self, vector_db, model, normalize=True):
        self.vector_db = vector_db if not normalize else self.normalize_vector(vector_db)
        self.model = model
        self.normalize = normalize
        

    def get_distances(self, data, input_format="text"):
        """
        Build the distance matrix between the query texts and the vector database.
        :param ls_texts: List of query texts
        :param vector_db: Vector database (encoded terms)
        :param model: SentenceTransformer model (the same used for vector_db)
        :param k: Number of nearest neighbors to retrieve
        :return: Distance matrix and indices of nearest neighbors
        """
        # k = k if k is not None else vector_db.shape[0]
        
        if input_format == "text":
            query_matrix = self.model.encode(data, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
        elif input_format == "vector":
            query_matrix = data if not self.normalize else self.normalize_vector(data)

        D = torch.mm(query_matrix, self.vector_db.T).cpu().numpy()
        I = D.argsort(axis=1)[:, ::-1]
        
        return D, I
    
    def get_top_k_gazetteer(self, df_gaz, D, I, k=10, code_col="code", term_col="term"):
        """
        Get the k closest terms for each query in the dataframe.
        """
        
        if k is None:
            k = D.shape[1]
            
        print("Getting the indices...")
        print("I shape:", I.shape)
        print("k", k)
        I_k = I[:, :k]
        print("Getting the distances...")
        D_k = D[np.arange(D.shape[0])[:, None], I_k]  # sorted distances
        
        print("Getting the terms and codes ...")

        ls_d_terms = []
        for i, row in enumerate(I_k):
            d_terms = {"codes": [], "terms": [], "similarity": []}
            for j, idx in enumerate(row):
                # print(i, j, idx)
                D_k[i][j] = D[i][idx]
                term = df_gaz.iloc[idx][[code_col, term_col]].to_dict()
                d_terms["codes"].append(term[code_col])
                d_terms["terms"].append(term[term_col])
                d_terms["similarity"].append(D_k[i][j].item())
            ls_d_terms.append(d_terms)
            
        return ls_d_terms
    
    def retrieve_top_k(self, data, df_gaz, k=10, input_format="text"):
        """
        Full retrieval function.
        :param data: List of query texts or query matrix
        :param vector_db: Vector database (encoded terms)
        :param model: SentenceTransformer model (the same used for vector_db)
        :param df_gaz: DataFrame with the gazetteer terms and codes
        :param k: Number of nearest neighbors to retrieve
        :return: Distance matrix and indices of nearest neighbors
        """
        
        D, I = self.get_distances(data, input_format=input_format)
        d_terms = self.get_top_k_gazetteer(df_gaz, D, I, k)
        
        return d_terms
    
    @staticmethod
    def normalize_vector(vector, p=2):
        """
        Normalize a vector to unit length.
        :param vector: Input vector
        :param p: Norm type (1 - taxi, 2 - euclidean, or np.inf - supremo)
        :return: Normalized vector
        """
        # norm = np.linalg.norm(vector, ord=p)
        norm = torch.norm(vector, p=p, dim=1, keepdim=True)
        if norm.sum() == 0:
            return vector
        return vector / norm