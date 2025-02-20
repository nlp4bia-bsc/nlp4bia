from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
from nlp4bia.datasets.utils import handlers

import os
        
from requests import get
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

class MeddoplaceLoader(BenchmarkDataset):
    URL = "https://zenodo.org/records/8403498/files/meddoplace_train+test+gazz+crossmap+multilingual_231003.zip?download=1"
    NAME = "meddoplace_train+test+gazz+crossmap+multilingual"
    DS_COLUMNS = config.DS_COLUMNS
    
    def __init__(self, lang="es", path=None, name=NAME, url=URL, download_if_missing=True, encoding="utf-8"):
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self):
        '''Load the data from the dataset
        Output: DataFrame with columns: filename, mark, label, off0, off1, span, code, semantic_rel, split, text
        '''
        
        train_path = os.path.join(self.path, "meddoplace_train/tsv/meddoplace_tsv_train_complete.tsv")
        texts_train_path = os.path.join(self.path, "meddoplace_train/txt")
        test_path = os.path.join(self.path, "meddoplace_test/tsv/meddoplace_tsv_test_complete.tsv")
        texts_test_path = os.path.join(self.path, "meddoplace_test/txt")

        df_train = pd.read_csv(train_path, sep="\t", dtype=str)
        df_test = pd.read_csv(test_path, sep="\t", dtype=str)
        
        df_train["split"] = "train"
        df_test["split"] = "test"
        
        df = pd.concat([df_train, df_test])
        df.rename(columns={"text": "span"}, inplace=True)
        
        df_texts = handlers.get_texts(texts_train_path, texts_test_path, encoding=self.encoding)
        df = df.merge(df_texts, on="filename", how="left")
        
        self.df = df
        
        return df
    
    def preprocess_data(self):
        print("preprocessing data...")
        # DS_COLUMNS =  ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]
        
        d_map_names = {"label": "mention_class", "need_context": "needs_context"}
        span_pos_cols = ["start_span", "end_span"]
        
        self.df["filenameid"] = self.df["filename"] + "#" + self.df[span_pos_cols[0]] + "#" + self.df[span_pos_cols[1]]
        self.df.drop(columns=["filename"] + span_pos_cols, inplace=True)

        self.df.rename(columns=d_map_names, inplace=True)
        
        for col in self.DS_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = None
        
        cols = self.DS_COLUMNS + ["text", "split"]
        self.df = self.df[cols]
        
        assert self.df.columns.intersection(self.DS_COLUMNS).shape[0] == len(self.DS_COLUMNS), "There are missing columns"
        
        return self.df
        
    def _download_data(self, download_path):
        # Ensure download path exists
        os.makedirs(download_path, exist_ok=True)

        # Download dataset
        print("Downloading dataset...")
        temp_zip_path = os.path.join(download_path, "temp_dataset.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract if zip file
        with ZipFile(temp_zip_path, 'r') as zip_file:
            zip_file.extractall(download_path)
        
        # Clean up the temporary zip file
        os.remove(temp_zip_path)
        print("Dataset downloaded and extracted successfully.")

        return download_path
            
class MeddoplaceGazetteer(BenchmarkDataset):
    URL = "https://zenodo.org/records/8403498/files/meddoplace_train+test+gazz+crossmap+multilingual_231003.zip?download=1"
    NAME = "meddoplace_train+test+gazz+crossmap+multilingual"
    DS_COLUMNS = config.DS_COLUMNS
    
    def __init__(self, lang="es", path=None, name=NAME, url=URL, download_if_missing=True, encoding="utf-8"):
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self):
        '''Load the data from the dataset
        Output: DataFrame with columns: filename, mark, label, off0, off1, span, code, semantic_rel, split, text
        '''
        gaz_path = os.path.join(self.path, "meddoplace_gazetteer/gazetteer_snomed_meddoplace.tsv")
        df = pd.read_csv(gaz_path, sep="\t", dtype=str, encoding=self.encoding)
        
        self.df = df
        
        return df
    
    def preprocess_data(self):
        print("preprocessing data...")
        # DS_COLUMNS =  ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]
        self.df = self.df[config.GZ_COLUMNS]
        
        return self.df
        
    def _download_data(self, download_path):
        # Ensure download path exists
        os.makedirs(download_path, exist_ok=True)

        # Download dataset
        print("Downloading dataset...")
        temp_zip_path = os.path.join(download_path, "temp_dataset.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract if zip file
        with ZipFile(temp_zip_path, 'r') as zip_file:
            zip_file.extractall(download_path)
        
        # Clean up the temporary zip file
        os.remove(temp_zip_path)
        print("Dataset downloaded and extracted successfully.")

        return download_path