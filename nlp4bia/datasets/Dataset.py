from abc import ABC, abstractmethod
import os
from nlp4bia.datasets import config

class AbstractDataset(ABC):
    @abstractmethod
    def load_data(self, indices=None):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass


class Dataset(AbstractDataset):
    ORIG_VERSION = config.ORIG_VERSION
    AVAIL_LANGS = ["en", "es", "fr", "de", "it", "nl", "pl", "pt", "ru", "tr"]
    
    def __init__(self, path, lang, version=None, download_if_missing=False, encoding="utf-8"):
        self.path = path
        self.d_versions = None
        self.lang = lang
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.encoding = encoding
        
        if self.lang not in self.AVAIL_LANGS:
            raise ValueError(f"Language '{self.lang}' is not supported for this dataset.")
                 
        self.version = version if version else self.get_latest_version()
        self.full_path = os.path.join(self.path, self.version)

    
    @abstractmethod
    def load_data(self):
        """Define in subclass"""
        pass
    
    @abstractmethod
    def preprocess_data(self):
        """Define in subclass"""
        pass
    
    def get_latest_version(self):
        versions = os.listdir(self.path)
        d_versions = {self.ORIG_VERSION: 0}
        v_versions = {v: int(v[1:]) for v in versions if v.startswith("v")}
        d_versions.update(v_versions)
        self.d_versions = dict(sorted(d_versions.items(), key=lambda item: item[1]))
        
        return list(self.d_versions.keys())[-1]
    
    def __repr__(self):
        return f"{type(self).__name__}(path={self.path}, version={self.version}, lang={self.lang})"

    def __str__(self):
        return f"{type(self).__name__}(path={self.path}, version={self.version}, lang={self.lang})"
    
class BenchmarkDataset(AbstractDataset):
    URL = None
    AVAIL_LANGS = ["en", "es", "fr", "de", "it", "nl", "pl", "pt", "ru", "tr"]
    
    def __init__(self, lang, name, path=None, url=None, download_if_missing=True, load=True, encoding="utf-8"):
        self.path = path if path is not None else os.path.join(config.NLP4BIA_DATA_PATH, name)
        self.name = name
        self.lang = lang
        self.df = None
        # self.df_train = None
        # self.df_val = None
        # self.df_test = None
        self.URL = url if url is not None else self.URL
        self.encoding = encoding
        
        if self.lang not in self.AVAIL_LANGS:
            raise ValueError(f"Language '{self.lang}' is not supported for this dataset.")
        
        if load:
            if not os.path.exists(self.path):
                if download_if_missing:
                    print(f"Path '{self.path}' does not exist. Downloading dataset to '{config.NLP4BIA_DATA_PATH}'...")
                    self._download_data(config.NLP4BIA_DATA_PATH)
                else:
                    raise FileNotFoundError(f"Path '{self.path}' does not exist, and download_if_missing is set to False.")
            
            self.load_data()
            self.preprocess_data()
        
        
    @abstractmethod
    def _download_data(self, download_path):
        # Placeholder for the dataset download logic
        os.makedirs(download_path, exist_ok=True)
        # Implement actual download code here, such as downloading from a URL
        print("Downloading dataset...")
        # Example: download dataset to download_path and return the path
        return download_path
    
    @abstractmethod
    def load_data(self):
        """Define in subclass"""
        pass
    
    @abstractmethod
    def preprocess_data(self):
        """Define in subclass"""
        pass
    
    
    def __repr__(self):
        return f"{type(self).__name__}(path={self.path}, url={self.URL}, lang={self.lang})"

    def __str__(self):
        return f"{type(self).__name__}(path={self.path}, url={self.URL}, lang={self.lang})"
    