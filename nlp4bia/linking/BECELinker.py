import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder

from nlp4bia.linking.retrievers import DenseRetriever
from nlp4bia.linking.rerankers import CrossEncoderReranker


class BECELinker:
    """
    A unified "Bi-Encoder + Cross-Encoder" entity linker.

    1) Uses a DenseRetriever (bi-encoder) to retrieve top-k candidates from a gazetteer.
    2) Uses a CrossEncoderReranker to re-score and sort those top-k candidates.

    Attributes:
        df_gazetteer (pd.DataFrame): The gazetteer DataFrame with "term" and "code" columns.
        retriever (DenseRetriever): Bi-encoder retriever instance.
        reranker (CrossEncoderReranker): Cross-encoder reranker instance.
    """

    def __init__(
        self,
        df_gazetteer: pd.DataFrame,
        biencoder_model_or_path: Union[SentenceTransformer, str],
        crossencoder_model_or_path: Union[CrossEncoder, str],
        normalize_embeddings: bool = True,
        biencoder_batch_size: int = 32,
        reranker_batch_size: int = 32,
        reranker_device: str = "cuda",
        biencoder_device: str = "cuda",
        show_progress_bar: bool = True
    ) -> None:
        """
        Initialize BECELinker with both bi-encoder and cross-encoder components.

        Args:
            df_gazetteer (pd.DataFrame):
                DataFrame containing the gazetteer. Must contain:
                  - "term": textual candidate
                  - "code": unique identifier for each term
            biencoder_model (SentenceTransformer):
                Pretrained SentenceTransformer instance for encoding queries and,
                if needed, encoding gazetteer terms.
            crossencoder_model_path (str):
                Path (or HuggingFace ID) to a pretrained CrossEncoder for reranking.
            normalize_embeddings (bool, optional):
                If True, L2-normalize gazetteer embeddings and query embeddings.
                Defaults to True.
            biencoder_batch_size (int, optional):
                Batch size for any encoding steps in the bi-encoder. Defaults to 32.
            reranker_batch_size (int, optional):
                Batch size for scoring pairs in the CrossEncoderReranker. Defaults to 32.
            retriever_device (str, optional):
                Device to load the bi-encoder model on (e.g. "cuda" or "cpu"). Defaults to "cuda".
            reranker_device (str, optional):
                Device to load the cross-encoder on. Defaults to "cuda".
            show_progress_bar (bool, optional):
                If True, show tqdm progress bars during encoding and scoring. Defaults to True.

        Raises:
            AssertionError: If `df_gazetteer` lacks the required columns "term" or "code".
        """
        assert "term" in df_gazetteer.columns, "`df_gazetteer` must contain a 'term' column"
        assert "code" in df_gazetteer.columns, "`df_gazetteer` must contain a 'code' column"
        self.df_gazetteer = df_gazetteer.reset_index(drop=True).copy()

        # Initialize DenseRetriever (bi-encoder)
        self.retriever = DenseRetriever(
            df_candidates=self.df_gazetteer,
            model_or_path=biencoder_model_or_path,
            normalize=normalize_embeddings,
            vector_db=None,  # will be computed inside DenseRetriever __init__
            vector_db_batch_size=biencoder_batch_size,
            device=biencoder_device
        )

        # Initialize CrossEncoderReranker
        term2code_mapping = self.df_gazetteer.set_index("term")["code"].to_dict()
        self.reranker = CrossEncoderReranker(
            model_or_path=crossencoder_model_or_path,
            device=reranker_device,
            batch_size=reranker_batch_size,
            term2code=term2code_mapping,
            show_progress_bar=show_progress_bar
        )

    def link(
        self,
        mentions: List[str],
        n_candidates: int = 200,
        top_k: int = 50,
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform entity linking for each mention by:
          1) Retrieving the top_k nearest neighbors from the gazetteer via bi-encoder.
          2) Re-ranking those k candidates with the cross-encoder.
          3) Returning the final sorted list for each mention.

        Args:
            mentions (List[str]):
                List of N mention strings to link.
            top_k (int, optional):
                Number of nearest neighbors to retrieve before reranking. Defaults to 50.
            return_documents (bool, optional):
                If True, each output dict includes "mention": the original query string.
                If False, omit that field. Defaults to True.

        Returns:
            List[Dict[str, Any]] of length N. Each dict contains:
              - (optional) "mention": str (only if return_documents=True)
              - "codes": List[str], final candidate codes sorted by cross-encoder score
              - "terms": List[str], final candidate terms sorted by score
              - "similarity": List[float], final cross-encoder scores (higher = better)
        """
        # Step 1: Retrieve top_k candidates via DenseRetriever
        raw_topk = self.retriever.retrieve_top_k(
            data=mentions,
            k=n_candidates,
            input_format="text",
            return_documents=return_documents
        )
        # raw_topk: List of dicts, each with keys:
        #   - "mention" (if return_documents=True)
        #   - "codes": List[str]
        #   - "terms": List[str]
        #   - "similarity": List[float]

        # Prepare candidate dicts for reranking: only "terms" & "codes" are needed
        cand_dicts: List[Dict[str, List[str]]] = []
        for entry in raw_topk:
            cand_dicts.append({
                "terms": entry["terms"],
                "codes": entry["codes"]
            })

        # Step 2: Rerank top_k candidates with CrossEncoder
        reranked = self.reranker.rerank(
            mentions=mentions,
            candidates=cand_dicts,
            k=top_k,  # keep all, but we could also pass top_k again if desired
            return_documents=return_documents
        )
        # reranked: List of dicts, each with keys:
        #   - "mention" (if return_documents=True)
        #   - "terms": sorted terms
        #   - "codes": sorted codes
        #   - "similarity": sorted cross-encoder scores

        return reranked