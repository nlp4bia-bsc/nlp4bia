from typing import List, Dict,Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import json
from meddoplace.misc.linking_structures import Mention
from meddoplace.misc.linking_structures import TermResult, Candidate
import requests
from tqdm import tqdm
import time

class BiencoderLinker:
    def __init__(self, model: SentenceTransformer, faiss_index):
        self.model = model
        self.faiss_index = faiss_index
    
    def get_candidates(self, terms: List[str], k: int = 5, add_synonyms = False) -> List[TermResult]:
        # Generar embeddings para los términos de entrada
        term_embeddings = self.model.encode(terms, show_progress_bar = True, convert_to_tensor=True)

        
        # Buscar los k vecinos más cercanos en el índice
        D, I = self.faiss_index.search(term_embeddings.cpu().numpy(), k)
        
        results = []
        for term_idx, term in enumerate(terms):
            candidates = {}
            for score_idx, score in enumerate(D[term_idx]):
                index = I[term_idx, score_idx]
                code = self.faiss_index.get_code_by_index(index)
                term_gazetteer = self.faiss_index.get_term_by_index(index)
                
                if code not in candidates or candidates[code].score < score:
                    if add_synonyms: 
                        gazz_df = self.faiss_index.gazetteer_df 
                        # Add to term all the synonyms of the code
                        term_gazetteer = "####".join(gazz_df[gazz_df.code==code].term.to_list())
                    candidates[code] = Candidate(code=code, score=score, term=term_gazetteer)
            
            # Convertir el diccionario de candidatos a una lista de objetos Candidate
            candidates_list = list(candidates.values())
            # TODO: Add method to include synonyms in the candidates "term". So
            # it appears like "term [SEP] sinonym [SEP] sinonym2". This method
            # might be useful for the maxsim step. We can use the self.faiss_index.gazetteer_df
            results.append(TermResult(term=term, candidates=candidates_list))
        
        return results

    def evaluate(self, gold_standard: List[Mention], candidates_pred: List[TermResult], k: int, 
                metrics: List[str] = ["acc@k"], output_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evalúa el reranker comparando el gold_standard con el output del BiencoderLinker tras ser rerankeado.
        """
        # # Aplicar el reranking sobre el output del BiencoderLinker
        # reranked_output = self.rerank(candidates_pred)

        # Inicializar diccionario de métricas
        results = {}

        if "acc@k" in metrics:
            # Diccionario para almacenar los resultados de acc@k
            acc_at_k_results = {}

            # Calcular acc@k para todos los valores desde 1 hasta k
            for i in range(1, k):
                acc_at_k = self.calculate_accuracy_at_k(gold_standard, candidates_pred, i)
                acc_at_k_results[i] = acc_at_k

            # Guardar los resultados de acc@k en el diccionario principal de resultados
            results['acc@k'] = acc_at_k_results

        # Calcular métricas adicionales
        if "f1-score" in metrics:
            flattened_true_codes = [gold.code for gold in gold_standard]
            flattened_pred_codes = [pred.candidates[0].code for pred in candidates_pred]

            f1 = f1_score(flattened_true_codes, flattened_pred_codes, average='macro')
            results["f1-score"] = f1

        # Si se proporciona output_path, guardar los resultados en un archivo JSON
        if output_path:
            with open(output_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)

        return results

    def calculate_accuracy_at_k(self, true_candidates: List[TermResult], pred_candidates: List[TermResult], k: int) -> float:
        import numpy as np
        
        return np.mean([any([true_term.code == pred_candidate.code for pred_candidate in pred_term.candidates[:k]]) for true_term, pred_term in zip(true_candidates, pred_candidates)])
