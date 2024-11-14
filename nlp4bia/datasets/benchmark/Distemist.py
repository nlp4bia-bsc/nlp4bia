from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
import os
        
from requests import get
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

class Distemist(BenchmarkDataset):
    URL = "https://zenodo.org/records/7614764/files/distemist_zenodo.zip?download=1"
    DS_COLUMNS = config.DS_COLUMNS
    
    def load_data(self):
        '''Load the data from the dataset
        Output: DataFrame with columns: filename, mark, label, off0, off1, span, code, semantic_rel, split, text
        '''
        
        train_path = os.path.join(self.path, "training/subtrack2_linking")
        texts_train_path = os.path.join(self.path, "training/text_files")
        test_path = os.path.join(self.path, "test_annotated/subtrack2_linking")
        texts_test_path = os.path.join(self.path, "test_annotated/text_files")
        
        df_train = pd.DataFrame()
        for path in os.listdir(train_path):
            df_i = pd.read_csv(os.path.join(train_path, path), sep="\t", dtype=str)
            df_train = pd.concat([df_train, df_i])
        
        df_test = pd.DataFrame()
        for path in os.listdir(test_path):
            df_i = pd.read_csv(os.path.join(test_path, path), sep="\t", dtype=str)
            df_test = pd.concat([df_test, df_i])
        
        df_train["split"] = "train"
        df_test["split"] = "test"
        
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        df_texts = self.get_texts(texts_train_path, texts_test_path)
        df = df.merge(df_texts, on="filename", how="left")
        
        assert df.duplicated(subset=["filename", "mark"]).sum() == 0, "There are duplicated filename+marks"
        
        # self.df_train = df[df["split"] == "train"]
        # self.df_test = df[df["split"] == "test"]
        self.df = df
        
        return df
    
    @staticmethod
    def get_texts(*paths, extension=".txt"):
        '''Get texts from text_files
        Input: paths: sequence of paths to text_files
        Output: DataFrame with columns: filename, text
        '''
        ls_texts_path = []
        for path in paths:
            ls_texts_path_i = []
            # For each main path, extract the filenames
            for filename in os.listdir(path):
                ls_texts_path_i.append((path, filename))
            
            # Append the list of filenames to the main list
            ls_texts_path.extend(ls_texts_path_i)
        
        # Retrieve the text from each file and create tuples with the filename and the content
        ls_texts = [(filename, open(os.path.join(path, filename)).read()) for (path, filename) in ls_texts_path]
        
        df_texts = pd.DataFrame(ls_texts, columns=["filename", "text"])
        df_texts["filename"] = df_texts["filename"].str.replace(extension, "") # remove the extension
                
        return df_texts
    
    def preprocess_data(self):
        print("preprocessing data...")
        # DS_COLUMNS =  ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]
        
        d_map_names = {
                        "filename": "filenameid",
                        "label": "mention_class",
                        "semantic_rel": "sem_rel",
        }
        
        self.df["filenameid"] = self.df["filename"] + "#" + self.df["off0"] + "#" + self.df["off1"]
        self.df.drop(columns=["filename", "off0", "off1"], inplace=True)
        
        self.df.rename(columns=d_map_names, inplace=True)
        
        for col in self.DS_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = None
        
        cols = self.DS_COLUMNS + ["text", "split"]
        self.df = self.df[cols]
        
    def _download_data(self, download_path):
        # Placeholder for the dataset download logic
        os.makedirs(download_path, exist_ok=True)
        # Implement actual download code here, such as downloading from a URL
        print("Downloading dataset...")
        # Example: download dataset to download_path and return the path
        # CACHE_DIR = os.path.join(DATASET_PATH, "cache")

        os.makedirs(download_path, exist_ok=True)
        response = get(self.URL)
        zip_file = ZipFile(BytesIO(response.content))
        zip_file.extractall(download_path)
        
        return download_path