import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
from tqdm import tqdm


class CrossEncoderReranker:
    """
    Uses a Sentence-Transformers CrossEncoder to re-score and sort a set of candidate terms for each mention.

    Attributes:
        model (CrossEncoder): Loaded CrossEncoder model used to score (mention, candidate) pairs.
        batch_size (int): Number of pairs to score in one forward pass.
        device (str): PyTorch device string for the CrossEncoder (e.g. "cuda" or "cpu").
        term2code (Optional[Dict[str, str]]): Optional mapping from candidate term → its code.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        term2code: Optional[Dict[str, str]] = None,
        show_progress_bar: bool = True
    ) -> None:
        """
        Instantiate the Reranker.

        Args:
            model_path (str):
                Path (or huggingface identifier) of a pretrained CrossEncoder. 
                E.g. "/path/to/crossencoder_model".
            device (str, optional):
                The torch device to load the model on. Defaults to "cuda".
            batch_size (int, optional):
                How many (mention, candidate) pairs to process in one batch. Defaults to 32.
            term2code (Dict[str, str], optional):
                If provided, a dictionary mapping each candidate term to its code. 
                If not provided, the caller must supply “codes” inside each candidate dict.
            show_progress_bar (bool, optional):
                Whether to show a tqdm progress bar during scoring. Defaults to True.
        """
        self.model = CrossEncoder(model_path, device=device)
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
        Flatten mentions & candidate-lists into one long list of (mention, candidate) pairs.
        Also record offsets so we can regroup scores per mention later.

        Args:
            mentions (List[str]):
                A list of N mention strings.
            candidates (List[Dict[str, List[str]]]):
                A list of length N, where each element is a dict with at least:
                  - "terms": List[str] of candidate term strings.
                  - (Optional) "codes": List[str] of codes corresponding to those terms.
                If "codes" is omitted, `self.term2code` must be provided so we can map term → code.

        Returns:
            all_pairs (List[Tuple[str, str]]):
                Flattened list of (mention, term) pairs to feed to CrossEncoder.
            offsets (List[Optional[Tuple[int, int]]]):
                For each mention i, offsets[i] = (start_idx, end_idx) in all_pairs such that
                all_pairs[start_idx:end_idx] are exactly the (mention, candidates[i]["terms"])
                pairs. If candidates[i]["terms"] is empty, offsets[i] = None.
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
        return_documents: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Given a list of mentions and a corresponding list of candidate-dicts,
        re-score all candidates with the CrossEncoder and return them sorted by descending score.

        Args:
            mentions (List[str]):
                A list of N mention strings.
            candidates (List[Dict[str, List[str]]]):
                A list of length N. Each element is a dict with:
                  - "terms": List[str] of candidate strings.
                  - (Optional) "codes": List[str] of codes for those terms.
                If "codes" is not provided, `self.term2code` must be non-None, and we'll look up codes by term.
            return_documents (bool, optional):
                If True, each output dict will include the original "mention" key.
                If False, the "mention" field is omitted. Defaults to True.

        Returns:
            reranked_results (List[Dict[str, Any]]):
                A list of length N. Each element is a dict with keys:
                  - (Optional) "mention": str, the original mention (only if return_documents=True)
                  - "terms": List[str], candidates sorted by score descending
                  - "codes": List[str], codes sorted in parallel with terms
                  - "scores": List[float], CrossEncoder scores (higher = better) sorted descending
        """
        if len(mentions) != len(candidates):
            raise ValueError("`mentions` and `candidates` must have the same length.")

        # Flatten into (mention, term) pairs and record offsets
        all_pairs, offsets = self._build_pair_list(mentions, candidates)

        # If there are no pairs at all, return empty structures
        if len(all_pairs) == 0:
            # Build an empty result for each mention
            empty_results = []
            for mention in mentions:
                entry: Dict[str, Any] = {
                    "terms": [],
                    "codes": [],
                    "scores": []
                }
                if return_documents:
                    entry["mention"] = mention
                empty_results.append(entry)
            return empty_results

        # CrossEncoder.predict accepts List[ (str, str) ]
        # It returns a numpy array of shape (len(all_pairs),)
        scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar
        )  # type: np.ndarray

        # Now regroup per mention using offsets, and sort each mention's candidates
        reranked_results: List[Dict[str, Any]] = []
        for i, mention in enumerate(mentions):
            offset = offsets[i]
            if offset is None:
                entry: Dict[str, Any] = {
                    "terms": [],
                    "codes": [],
                    "scores": []
                }
                if return_documents:
                    entry["mention"] = mention
                reranked_results.append(entry)
                continue

            start, end = offset
            slice_scores = scores[start:end]  # shape = (num_cands_for_mention,)
            term_list = candidates[i]["terms"]

            # Determine codes: either from candidate dict or via self.term2code
            if "codes" in candidates[i]:
                code_list = candidates[i]["codes"]
                if len(code_list) != len(term_list):
                    raise ValueError(
                        f"For mention {i}, 'terms' and 'codes' lists must be same length."
                    )
            else:
                if self.term2code is None:
                    raise ValueError(
                        "No 'codes' in candidate dict and no term2code mapping provided in Reranker."
                    )
                code_list = []
                for t in term_list:
                    if t not in self.term2code:
                        raise KeyError(f"Term '{t}' not found in term2code mapping.")
                    code_list.append(self.term2code[t])

            # Sort indices by descending score
            sorted_indices = np.argsort(slice_scores)[::-1]
            sorted_terms = [term_list[idx] for idx in sorted_indices]
            sorted_codes = [code_list[idx] for idx in sorted_indices]
            sorted_scores = [float(slice_scores[idx]) for idx in sorted_indices]

            entry: Dict[str, Any] = {
                "terms": sorted_terms,
                "codes": sorted_codes,
                "scores": sorted_scores
            }
            if return_documents:
                entry["mention"] = mention

            reranked_results.append(entry)

        return reranked_results
