import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """
    Performs dense retrieval of query embeddings against a precomputed vector database
    of candidate terms (the gazetteer). 

    The gazetteer is stored as a DataFrame (`df_candidates`) containing at least two columns:
      - "term": the string form of each candidate
      - "code": a unique identifier (or code) corresponding to each term

    This class can either accept a precomputed embedding matrix (`vector_db`) for the gazetteer,
    or compute it on-the-fly by encoding the "term" column with a provided SentenceTransformer model.

    Attributes:
        df_candidates (pd.DataFrame): Copy of the input DataFrame with columns "term" and "code".
        vector_db (torch.Tensor): Tensor of shape (num_terms, embedding_dim) containing
                                  (optionally normalized) gazetteer embeddings.
        model (SentenceTransformer): Model used to encode queries (if input_format="text").
        normalize (bool): Whether to L2-normalize incoming query vectors (or provided `vector_db`).
    """

    def __init__(
        self,
        df_candidates: pd.DataFrame,
        model_or_path: Union[str, SentenceTransformer],
        normalize: bool = True,
        vector_db: Optional[torch.Tensor] = None,
        vector_db_batch_size: int = 256,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        """
        Initialize a DenseRetriever over a candidate-term DataFrame.

        Args:
            df_candidates (pd.DataFrame):
                DataFrame containing the gazetteer. Must contain columns:
                  - "term": the textual form of each candidate
                  - "code": a unique identifier or code for each candidate
            model_or_path (str or SentenceTransformer):
                Either a string path/identifier for a pretrained SentenceTransformer
                or an already-loaded SentenceTransformer instance. If a path is given,
                the model will be loaded from that path.
            normalize (bool, optional):
                If True:
                  - L2-normalize the rows of `vector_db` (if provided).
                  - When encoding queries (input_format="text"), automatically
                    normalize the query embeddings to unit length.
                Defaults to True.
            vector_db (torch.Tensor, optional):
                If provided, a tensor of shape (num_terms, embedding_dim) containing
                precomputed embeddings for all rows in `df_candidates["term"]`. If None,
                this constructor will encode `df_candidates["term"]` via `model.encode(...)`.
                If this tensor is supplied and `normalize=True`, it will be L2-normalized
                across dim=1. Defaults to None.
            vector_db_batch_size (int, optional):
                Batch size to use when encoding the gazetteer terms (only when
                `vector_db` is None). Defaults to 256.
            device (str or torch.device, optional):
                The device to use for computations. Can be "cpu" or "cuda".
                Defaults to "cuda" if available, otherwise "cpu".

        Raises:
            AssertionError:
                If `df_candidates` does not contain the required columns "term" and "code".
        """
        self.normalize = normalize
        self.df_candidates = df_candidates.copy()
        assert "term" in self.df_candidates.columns, "`df_candidates` must contain a 'term' column"
        assert "code" in self.df_candidates.columns, "`df_candidates` must contain a 'code' column"
        self.device = device
        
        # Load or assign the SentenceTransformer model
        if isinstance(model_or_path, SentenceTransformer):
            self.model = model_or_path
        else:
            self.model = SentenceTransformer(model_or_path, device=device)

        if vector_db is None:
            # Compute gazetteer embeddings by encoding each term string
            # The result is a tensor of shape (num_terms, embedding_dim).
            self.vector_db: torch.Tensor = self.model.encode(
                self.df_candidates["term"].tolist(),
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                batch_size=vector_db_batch_size,
                device=self.device
            )
        else:
            # Use the supplied embedding matrix; optionally normalize each row
            self.vector_db = (
                self.normalize_vector(vector_db) if self.normalize else vector_db
            )

    def get_distances(
        self,
        data: Union[List[str], torch.Tensor],
        input_format: str = "text"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine-similarity scores between each query and all gazetteer embeddings.

        This method returns two arrays:
          1. `distances`: a NumPy array of shape (num_queries, num_terms) containing
             cosine-similarity scores between each query and each gazetteer entry.
          2. `indices`: a NumPy array of shape (num_queries, num_terms) containing,
             for each query-row, the gazetteer indices sorted in descending order of similarity.

        Args:
            data (List[str] or torch.Tensor):
                - If `input_format="text"`: a list of raw query strings. Each string
                  will be encoded by `self.model.encode(...)`, resulting in a tensor of
                  shape (num_queries, embedding_dim).
                - If `input_format="vector"`: a torch.Tensor of shape
                  (num_queries, embedding_dim) containing precomputed query embeddings.
            input_format (str, optional):
                Specifies how to interpret `data`:
                  - "text": treat `data` as List[str] and encode with `self.model`.
                            The encoded queries are L2-normalized if `self.normalize=True`.
                  - "vector": treat `data` as a precomputed embedding tensor. If
                              `self.normalize=True`, these query embeddings are L2-normalized.
                Default is "text".

        Returns:
            distances (np.ndarray):
                Cosine-similarity scores with shape (num_queries, num_terms).
                Higher values indicate greater similarity.
            indices (np.ndarray):
                For each query (row), an array of length num_terms giving the
                gazetteer-entry indices sorted by descending similarity.

        Raises:
            ValueError:
                If `input_format` is not one of {"text", "vector"}.
        """
        if input_format == "text":
            query_matrix: torch.Tensor = self.model.encode(
                data,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                device=self.device
            )
        elif input_format == "vector":
            raw_queries: torch.Tensor = data  # type: ignore
            query_matrix = (
                self.normalize_vector(raw_queries) if self.normalize else raw_queries
            )
        else:
            raise ValueError(f"input_format must be 'text' or 'vector', got '{input_format}'")

        similarity_tensor: torch.Tensor = torch.mm(query_matrix, self.vector_db.T)
        distances: np.ndarray = similarity_tensor.cpu().numpy()
        indices: np.ndarray = distances.argsort(axis=1)[:, ::-1]
        return distances, indices

    def get_top_k_gazetteer(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        k: Optional[int] = 10
    ) -> List[Dict[str, List[Union[str, float]]]]:
        """
        Retrieve the top-k nearest neighbors from the gazetteer for each query.

        Given `distances` and `indices` (as returned by `get_distances`), this method
        builds, for each query, a dictionary containing:
          - "codes": the gazetteer codes of the top-k entries
          - "terms": the gazetteer term strings of the top-k entries
          - "similarity": the cosine-similarity scores of those top-k entries

        Args:
            distances (np.ndarray):
                Cosine-similarity scores array of shape (num_queries, num_terms).
            indices (np.ndarray):
                Sorted gazetteer indices by descending similarity (shape (num_queries, num_terms)).
            k (int or None, optional):
                Number of nearest neighbors to retrieve for each query.
                If k is None or exceeds `num_terms`, returns all `num_terms`.
                Default is 10.

        Returns:
            List[Dict[str, List[Union[str, float]]]]:
                A list of length num_queries, where each element is a dict with keys:
                  - "codes": List[str] of length k (gazetteer codes)
                  - "terms": List[str] of length k (gazetteer term strings)
                  - "similarity": List[float] of length k (cosine-similarity scores)
        """
        num_queries, num_terms = distances.shape
        if k is None or k > num_terms:
            k = num_terms

        topk_indices: np.ndarray = indices[:, :k]
        topk_similarities: np.ndarray = distances[np.arange(num_queries)[:, None], topk_indices]

        top_k_list: List[Dict[str, List[Union[str, float]]]] = []
        codes_list = self.df_candidates["code"].tolist()
        terms_list = self.df_candidates["term"].tolist()

        for qi in range(num_queries):
            entry = {"codes": [], "terms": [], "similarity": []}
            for rank_idx, gaz_idx in enumerate(topk_indices[qi]):
                entry["codes"].append(str(codes_list[gaz_idx]))
                entry["terms"].append(str(terms_list[gaz_idx]))
                entry["similarity"].append(float(topk_similarities[qi, rank_idx]))
            top_k_list.append(entry)

        return top_k_list

    def retrieve_top_k(
        self,
        data: Union[List[str], torch.Tensor],
        k: int = 10,
        input_format: str = "text",
        return_documents: bool = True
    ) -> List[Dict[str, List[Union[str, float]]]]:
        """
        End-to-end retrieval: encode queries (if needed), compute similarity against
        all gazetteer entries, and return the top-k neighbors (with codes, terms, similarity).
        Optionally attaches the original query string under the key "mention".

        Args:
            data (List[str] or torch.Tensor):
                - If `input_format="text"`: a list of raw query strings to encode.
                - If `input_format="vector"`: a torch.Tensor of shape (num_queries, embedding_dim)
                  of precomputed query embeddings.
            k (int, optional):
                Number of nearest neighbors to return per query. Default is 10.
            input_format (str, optional):
                One of {"text", "vector"}:
                  - "text": treat `data` as List[str], encode with `self.model.encode(...)`.
                  - "vector": treat `data` as a tensor of query embeddings.
                Default is "text".
            return_documents (bool, optional):
                If True and `input_format=="text"`, each returned dict will include
                a key `"mention"` set to the original query string. If False, omit it.
                Defaults to True.

        Returns:
            List[Dict[str, List[Union[str, float]]]]:
                A list of length num_queries. Each element is a dictionary with keys:
                  - "codes": List[str] of length k
                  - "terms": List[str] of length k
                  - "similarity": List[float] of length k
                  - (optional) "mention": str (only if return_documents=True and input_format="text")

        Raises:
            ValueError:
                If `input_format` is not "text" or "vector".
        """
        distances, indices = self.get_distances(data, input_format=input_format)
        top_k_results = self.get_top_k_gazetteer(distances, indices, k)

        if input_format == "text" and return_documents:
            for i, query_str in enumerate(data):
                top_k_results[i]["mention"] = query_str

        return top_k_results

    @staticmethod
    def normalize_vector(
        vector: torch.Tensor,
        p: Union[int, float] = 2
    ) -> torch.Tensor:
        """
        L2-normalize (or L-p normalize) each row of the input tensor.

        Args:
            vector (torch.Tensor):
                Input tensor of shape (N, D), where N = number of vectors (rows),
                D = embedding dimension (columns).
            p (int or float, optional):
                The norm degree:
                  - p=1: L1 norm
                  - p=2: L2 norm (default)
                  - p=np.inf: max-norm
                Default is 2 (L2 norm).

        Returns:
            torch.Tensor:
                Tensor of shape (N, D) where each row has been divided by its L-p norm.
                If any row has zero norm, that row remains unchanged (division by 1).
        """
        norm = torch.norm(vector, p=p, dim=1, keepdim=True)
        zero_rows = (norm == 0).squeeze(1)
        if zero_rows.any():
            norm[zero_rows, :] = 1.0
        return vector / norm
