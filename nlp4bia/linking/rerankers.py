import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class CrossEncoderReranker:
    """
    Uses a Sentence-Transformers CrossEncoder to re‐score and sort a set of candidate
    terms for each mention.

    Attributes:
        model (CrossEncoder): CrossEncoder instance for scoring (mention, candidate) pairs.
        batch_size (int): Number of pairs to score in one forward pass.
        device (str): PyTorch device for the CrossEncoder (e.g., "cuda" or "cpu").
        term2code (Optional[Dict[str, str]]): Mapping from candidate term → code.
    """

    def __init__(
        self,
        model_or_path: Union[str, CrossEncoder],
        device: str = "cuda",
        batch_size: int = 32,
        term2code: Optional[Dict[str, str]] = None,
        show_progress_bar: bool = True
    ) -> None:
        """
        Instantiate the reranker.

        Args:
            model_or_path (str or CrossEncoder):
                Either a string path/identifier for a pretrained CrossEncoder (e.g.
                "/path/to/crossencoder_model") or a pre-instantiated CrossEncoder object.
                If a string is provided, the CrossEncoder will be loaded on `device`.
            device (str, optional):
                Torch device for loading the model (ignored if `model_or_path` is already
                a CrossEncoder). Defaults to "cuda".
            batch_size (int, optional):
                How many (mention, candidate) pairs to process per forward pass. Defaults to 32.
            term2code (Dict[str, str], optional):
                If provided, a dict mapping each candidate term to its code. If not provided,
                each candidate dict must include a "codes" list.
            show_progress_bar (bool, optional):
                Whether to show a tqdm progress bar during scoring. Defaults to True.
        """
        if isinstance(model_or_path, CrossEncoder):
            self.model = model_or_path
        else:
            self.model = CrossEncoder(model_or_path, device=device)

        self.device = device
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.term2code = term2code

    def _build_pair_list(
        self,
        mentions: List[str],
        candidates: List[Dict[str, List[str]]]
    ) -> Tuple[List[Tuple[str, str]], List[Optional[Tuple[int, int]]]]:
        """
        Flatten mentions & candidate-lists into a long list of (mention, candidate) pairs,
        and record offsets to regroup scores per mention.

        Args:
            mentions (List[str]):
                A list of N mention strings.
            candidates (List[Dict[str, List[str]]]):
                A list of length N; each element is a dict with:
                  - "terms": List[str] of candidate strings.
                  - (Optional) "codes": List[str] of codes for those terms.
                If "codes" is omitted, `self.term2code` must be provided.

        Returns:
            all_pairs (List[Tuple[str, str]]):
                Flattened list of (mention, term) tuples to feed to the CrossEncoder.
            offsets (List[Optional[Tuple[int, int]]]):
                For each mention i, offsets[i] = (start_idx, end_idx) in `all_pairs`
                corresponding to that mention’s candidates; or None if no candidates.
        """
        all_pairs: List[Tuple[str, str]] = []
        offsets: List[Optional[Tuple[int, int]]] = [None] * len(mentions)
        cursor = 0

        for i, (mention, cand_dict) in enumerate(zip(mentions, candidates)):
            terms = cand_dict.get("terms", [])
            if not terms:
                offsets[i] = None
                continue

            start = cursor
            for term in terms:
                all_pairs.append((mention, term))
                cursor += 1
            end = cursor
            offsets[i] = (start, end)

        return all_pairs, offsets

    def rerank(
        self,
        mentions: List[str],
        candidates: List[Dict[str, List[str]]],
        k: int = 10,
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Given a list of mentions and their candidate dicts, re‐score all candidates with
        the CrossEncoder and return them sorted by descending score.

        Args:
            mentions (List[str]):
                List of N mention strings.
            candidates (List[Dict[str, List[str]]]):
                List of N dicts, each with:
                  - "terms": List[str] of candidate strings.
                  - (Optional) "codes": List[str] of codes for those terms.
                If "codes" is omitted, `self.term2code` must be provided.
            k (int, optional):
                Return only the top‐k candidates per mention after reranking. Defaults to 10.
            return_documents (bool, optional):
                If True, each output dict includes "mention": the original mention string.
                Defaults to True.

        Returns:
            List[Dict[str, Any]]:
                A list of N dicts, each with keys:
                  - (Optional) "mention": str (if return_documents=True)
                  - "terms": List[str], top‐k candidates sorted by score
                  - "codes": List[str], corresponding codes sorted by score
                  - "similarity": List[float], CrossEncoder scores sorted descending
        """
        if len(mentions) != len(candidates):
            raise ValueError("`mentions` and `candidates` must have the same length.")

        all_pairs, offsets = self._build_pair_list(mentions, candidates)

        if not all_pairs:
            empty_results = []
            for mention in mentions:
                entry: Dict[str, Any] = {"terms": [], "codes": [], "similarity": []}
                if return_documents:
                    entry["mention"] = mention
                empty_results.append(entry)
            return empty_results

        scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar
        )  # shape: (len(all_pairs),)

        reranked_results: List[Dict[str, Any]] = []
        for i, mention in enumerate(mentions):
            offset = offsets[i]
            if offset is None:
                entry: Dict[str, Any] = {"terms": [], "codes": [], "similarity": []}
                if return_documents:
                    entry["mention"] = mention
                reranked_results.append(entry)
                continue

            start, end = offset
            slice_scores = scores[start:end]
            term_list = candidates[i]["terms"]

            if "codes" in candidates[i]:
                code_list = candidates[i]["codes"]
                if len(code_list) != len(term_list):
                    raise ValueError(
                        f"For mention {i}, 'terms' and 'codes' lists must be the same length."
                    )
            else:
                if self.term2code is None:
                    raise ValueError(
                        "No 'codes' in candidate dict and no term2code mapping provided."
                    )
                code_list = [self.term2code[t] for t in term_list]

            sorted_indices = np.argsort(slice_scores)[::-1]
            sorted_terms = [term_list[idx] for idx in sorted_indices][:k]
            sorted_codes = [code_list[idx] for idx in sorted_indices][:k]
            sorted_scores = [float(slice_scores[idx]) for idx in sorted_indices][:k]

            entry: Dict[str, Any] = {
                "terms": sorted_terms,
                "codes": sorted_codes,
                "similarity": sorted_scores
            }
            if return_documents:
                entry["mention"] = mention

            reranked_results.append(entry)

        return reranked_results
