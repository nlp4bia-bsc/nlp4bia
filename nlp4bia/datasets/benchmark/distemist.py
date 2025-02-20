from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
from nlp4bia.datasets.utils import handlers
import os
        
from requests import get
from zipfile import ZipFile
from io import BytesIO
import pandas as pd

class DistemistLoader(BenchmarkDataset):
    URL = "https://zenodo.org/records/7614764/files/distemist_zenodo.zip?download=1"
    NAME = "distemist_zenodo"
    DS_COLUMNS = config.DS_COLUMNS
    
    def __init__(self, lang="es", path=None, name=NAME, url=URL, download_if_missing=True, encoding="utf-8"):
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

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
        
        df_texts = handlers.get_texts(texts_train_path, texts_test_path, encoding=self.encoding)
        df = df.merge(df_texts, on="filename", how="left")
        
        assert df.duplicated(subset=["filename", "mark"]).sum() == 0, "There are duplicated filename+marks"
        
        self.df = df
        
        return df
    
    # @staticmethod
    # def get_texts(*paths, extension=".txt", encoding="utf-8"):
    #     '''Get texts from text_files
    #     Input: paths: sequence of paths to text_files
    #     Output: DataFrame with columns: filename, text
    #     '''
    #     ls_texts_path = []
    #     for path in paths:
    #         ls_texts_path_i = []
    #         # For each main path, extract the filenames
    #         for filename in os.listdir(path):
    #             ls_texts_path_i.append((path, filename))
            
    #         # Append the list of filenames to the main list
    #         ls_texts_path.extend(ls_texts_path_i)
        
    #     # Retrieve the text from each file and create tuples with the filename and the content
    #     ls_texts = [(filename, open(os.path.join(path, filename), encoding=encoding).read()) for (path, filename) in ls_texts_path]
        
    #     df_texts = pd.DataFrame(ls_texts, columns=["filename", "text"])
    #     df_texts["filename"] = df_texts["filename"].str.replace(extension, "") # remove the extension
    #     return df_texts
    
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
        
        assert self.df.columns.intersection(self.DS_COLUMNS).shape[0] == len(self.DS_COLUMNS), "There are missing columns"
        
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
    
class DistemistGazetteer(BenchmarkDataset):
    URL = "https://zenodo.org/records/6505583/files/dictionary_distemist.tsv?download=1"
    NAME = "dictionary_distemist"
    
    def __init__(self, lang="es", path=None, name=NAME, url=URL, download_if_missing=True, encoding="utf-8"):
        super().__init__(lang, name, path, url, download_if_missing, load=False, encoding=encoding)

        self.path = os.path.join(self.path, self.NAME + ".tsv")
        
        if not os.path.exists(self.path):
            if download_if_missing:
                print(f"Path '{self.path}' does not exist. Downloading dataset to '{self.path}'...")
                self._download_data(self.path)
            else:
                raise FileNotFoundError(f"Path '{self.path}' does not exist, and download_if_missing is set to False.")
        
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        # Ensuring self.filename path consistency
        # self.filename = os.path.join(self.path, self.NAME + ".tsv")
        self.df = pd.read_csv(self.path, sep="\t", dtype=str, encoding=self.encoding)
    
    def preprocess_data(self):
        print("preprocessing data...")
        # DS_COLUMNS =  ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]
        self.df = self.df[config.GZ_COLUMNS]
        
        return self.df

        
    def _download_data(self, download_path):
        # Ensure the directory for download_path exists
        print("Downloading dataset...")
        response = get(self.URL)
        
        os.makedirs(os.path.dirname(download_path), exist_ok=True)          
            
        # Save the response content directly as a .tsv file
        with open(download_path, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded dataset saved to {download_path}")
        return download_path