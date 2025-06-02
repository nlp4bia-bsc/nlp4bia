import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """
    Performs dense retrieval of query embeddings against a precomputed vector database (gazetteer).

    Attributes:
        vector_db (torch.Tensor): Normalized or raw embeddings of gazetteer terms.
        model (SentenceTransformer): SentenceTransformer model used to encode query texts.
        normalize (bool): Whether to normalize input vectors before retrieval.
    """

    def __init__(
        self,
        vector_db: torch.Tensor,
        model: SentenceTransformer,
        normalize: bool = True
    ) -> None:
        """
        Initialize the DenseRetriever.

        Args:
            vector_db (torch.Tensor):
                Precomputed embeddings for gazetteer terms. Shape: (num_terms, embedding_dim).
            model (SentenceTransformer):
                SentenceTransformer instance used to encode query texts.
            normalize (bool, optional):
                If True, the provided `vector_db` will be L2-normalized along dim=1. Default is True.
        """
        self.normalize = normalize
        self.vector_db: torch.Tensor = (
            self.normalize_vector(vector_db) if normalize else vector_db
        )
        self.model: SentenceTransformer = model

    def get_distances(
        self,
        data: Union[List[str], torch.Tensor],
        input_format: str = "text"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cosine similarity scores between query embeddings and the vector database.

        If `input_format="text"`, the `data` argument should be a list of raw query strings,
        which will be encoded via `self.model.encode(...)`. If `input_format="vector"`, then
        `data` is assumed to be a tensor of precomputed query embeddings.

        Args:
            data (List[str] or torch.Tensor):
                - If input_format == "text": list of raw query strings.
                - If input_format == "vector": tensor of shape (num_queries, embedding_dim).
            input_format (str, optional):
                - "text": encode `data` as raw strings using `self.model`.
                - "vector": use `data` directly as embeddings (will normalize if `self.normalize=True`).
                Default is "text".

        Returns:
            distances (np.ndarray):
                Cosine-similarity scores between each query and all gazetteer embeddings.
                Shape: (num_queries, num_terms).
            indices (np.ndarray):
                Indices of gazetteer entries sorted by descending similarity for each query.
                Shape: (num_queries, num_terms).
        """
        # Encode or normalize query embeddings
        if input_format == "text":
            # Returns tensor of shape (num_queries, embedding_dim)
            query_matrix: torch.Tensor = self.model.encode(
                data,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        elif input_format == "vector":
            # If provided as vectors, optionally normalize
            raw_queries: torch.Tensor = data  # type: ignore
            query_matrix = (
                self.normalize_vector(raw_queries) if self.normalize else raw_queries
            )
        else:
            raise ValueError(f"input_format must be 'text' or 'vector', got '{input_format}'")

        # Compute cosine similarity via matrix multiplication
        # self.vector_db is shape (num_terms, embedding_dim)
        # query_matrix shape: (num_queries, embedding_dim)
        # Hence, torch.mm(query_matrix, self.vector_db.T) -> (num_queries, num_terms)
        similarity_tensor: torch.Tensor = torch.mm(query_matrix, self.vector_db.T)

        # Move to CPU numpy arrays
        distances: np.ndarray = similarity_tensor.cpu().numpy()

        # Sort indices in descending order of similarity
        # argsort returns ascending, so we reverse with [ :, ::-1 ]
        indices: np.ndarray = distances.argsort(axis=1)[:, ::-1]

        return distances, indices

    def get_top_k_gazetteer(
        self,
        df_gaz: pd.DataFrame,
        distances: np.ndarray,
        indices: np.ndarray,
        k: Optional[int] = 10,
        code_col: str = "code",
        term_col: str = "term"
    ) -> List[Dict[str, List[Union[str, float]]]]:
        """
        Retrieve the top-k closest gazetteer entries (codes and terms) for each query.

        Args:
            df_gaz (pd.DataFrame):
                DataFrame containing the gazetteer, with at least columns `code_col` and `term_col`.
            distances (np.ndarray):
                Cosine-similarity scores array of shape (num_queries, num_terms).
            indices (np.ndarray):
                Sorted gazetteer indices by descending similarity, shape (num_queries, num_terms).
            k (int or None, optional):
                Number of top nearest neighbors to retrieve. If None, returns all. Default is 10.
            code_col (str, optional):
                Column name in `df_gaz` for the gazetteer code. Default is "code".
            term_col (str, optional):
                Column name in `df_gaz` for the gazetteer term. Default is "term".

        Returns:
            top_k_list (List[Dict[str, List[Union[str, float]]]]):
                A list (one per query) of dictionaries with keys:
                  - "codes": List[str] of length k
                  - "terms": List[str] of length k
                  - "similarity": List[float] of corresponding cosine scores
        """
        num_queries, num_terms = distances.shape

        # If k is None or larger than available terms, use all terms
        if k is None or k > num_terms:
            k = num_terms

        # For each query, take the first k indices by highest similarity
        topk_indices: np.ndarray = indices[:, :k]

        # Gather distances of the top-k entries for each query
        # distances[np.arange(num_queries)[:, None], topk_indices] -> shape (num_queries, k)
        topk_distances: np.ndarray = distances[np.arange(num_queries)[:, None], topk_indices]

        top_k_list: List[Dict[str, List[Union[str, float]]]] = []

        for query_idx in range(num_queries):
            # Prepare containers for codes, terms, and similarity scores
            entry = {"codes": [], "terms": [], "similarity": []}

            for rank_idx, gaz_idx in enumerate(topk_indices[query_idx]):
                code_value: str = str(df_gaz.iloc[gaz_idx][code_col])
                term_value: str = str(df_gaz.iloc[gaz_idx][term_col])
                sim_score: float = float(topk_distances[query_idx, rank_idx])

                entry["codes"].append(code_value)
                entry["terms"].append(term_value)
                entry["similarity"].append(sim_score)

            top_k_list.append(entry)

        return top_k_list

    def retrieve_top_k(
        self,
        data: Union[List[str], torch.Tensor],
        df_gaz: pd.DataFrame,
        k: int = 10,
        input_format: str = "text"
    ) -> List[Dict[str, List[Union[str, float]]]]:
        """
        Perform full retrieval: encode queries, compute similarity to the gazetteer, and return top-k.

        Args:
            data (List[str] or torch.Tensor):
                If `input_format=="text"`, a list of raw query strings.
                If `input_format=="vector"`, a tensor of shape (num_queries, embedding_dim).
            df_gaz (pd.DataFrame):
                DataFrame containing the gazetteer entries. Must have `code` and `term` columns,
                or specified via `code_col`/`term_col` arguments in get_top_k_gazetteer.
            k (int, optional):
                Number of nearest neighbors to return. Default is 10.
            input_format (str, optional):
                Format of `data`. "text" for raw strings, "vector" for precomputed embeddings.
                Default is "text".

        Returns:
            List[Dict[str, List[Union[str, float]]]]:
                A list (length = num_queries) of dicts, each containing:
                  - "codes": List[str]
                  - "terms": List[str]
                  - "similarity": List[float]
        """
        # Step 1: compute distances and sorted indices
        distances, indices = self.get_distances(data, input_format=input_format)

        # Step 2: extract top-k gazetteer entries
        top_k_results = self.get_top_k_gazetteer(df_gaz, distances, indices, k)

        return top_k_results

    @staticmethod
    def normalize_vector(
        vector: torch.Tensor,
        p: Union[int, float] = 2
    ) -> torch.Tensor:
        """
        Normalize each row of the input tensor to unit length using the L-p norm.

        Args:
            vector (torch.Tensor):
                Input tensor of shape (N, D), where N = number of vectors, D = embedding dimension.
            p (int or float, optional):
                The norm degree (1 for L1, 2 for L2, np.inf for max-norm). Default is 2.

        Returns:
            torch.Tensor:
                Normalized tensor of shape (N, D). If any row has zero norm, it remains unchanged.
        """
        # Compute the norm along dimension 1 (each row), keepdims for broadcasting
        norm = torch.norm(vector, p=p, dim=1, keepdim=True)  # shape: (N, 1)

        # Avoid division by zero: if a row's norm is zero, keep the original row
        zero_rows = (norm == 0).squeeze(1)  # shape: (N,)
        if zero_rows.any():
            # For zero-norm rows, set norm to 1 before division to leave them unchanged
            norm[zero_rows, :] = 1.0

        return vector / norm
