import os
import glob
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from requests import get

from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
from nlp4bia.datasets.utils import handlers


class DistemistLoader(BenchmarkDataset):
    """
    Loader for the Distemist dataset (disease mentions recognition & normalization in Spanish medical texts).

    Inherits from BenchmarkDataset, which handles download and basic path management.

    Attributes:
        URL (str): Download URL for the Distemist ZIP archive.
        NAME (str): Name of the dataset folder once extracted.
        DS_COLUMNS (List[str]): Expected columns for the dataset after preprocessing.
    """

    URL: str = "https://zenodo.org/records/7614764/files/distemist_zenodo.zip?download=1"
    NAME: str = "distemist_zenodo"
    DS_COLUMNS = config.DS_COLUMNS  # e.g., ["filenameid", "mention_class", "span", "code", "sem_rel", "is_abbreviation", "is_composite", "needs_context", "extension_esp"]

    def __init__(
        self,
        lang: str = "es",
        path: Optional[str] = None,
        name: str = NAME,
        url: str = URL,
        download_if_missing: bool = True,
        encoding: str = "utf-8"
    ) -> None:
        """
        Initialize the DistemistLoader.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the dataset. If None, uses default in BenchmarkDataset.
            name (str, optional): Dataset folder name (default is NAME).
            url (str, optional): URL to download the Distemist ZIP file.
            download_if_missing (bool, optional): If True, downloads data when not present locally.
            encoding (str, optional): File encoding for reading TSV/CSV (default "utf-8").
        """
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self) -> pd.DataFrame:
        """
        Load Distemist train and test annotations, merge them, and join with raw text.

        Expects the following folder structure under self.path after extraction:
            - training/subtrack2_linking/
            - training/text_files/
            - test_annotated/subtrack2_linking/
            - test_annotated/text_files/

        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                ["filename", "mark", "label", "off0", "off1", "span", "code", "semantic_rel", "split", "text"].
        """
        # Define paths to annotation folders
        train_ann_dir = os.path.join(self.path, "training", "subtrack2_linking")
        train_text_dir = os.path.join(self.path, "training", "text_files")
        test_ann_dir = os.path.join(self.path, "test_annotated", "subtrack2_linking")
        test_text_dir = os.path.join(self.path, "test_annotated", "text_files")

        # Load and concatenate all train annotation TSVs
        df_train = pd.DataFrame()
        for filename in os.listdir(train_ann_dir):
            file_path = os.path.join(train_ann_dir, filename)
            df_part = pd.read_csv(file_path, sep="\t", dtype=str, encoding=self.encoding)
            df_train = pd.concat([df_train, df_part], ignore_index=True)

        # Load and concatenate all test annotation TSVs
        df_test = pd.DataFrame()
        for filename in os.listdir(test_ann_dir):
            file_path = os.path.join(test_ann_dir, filename)
            df_part = pd.read_csv(file_path, sep="\t", dtype=str, encoding=self.encoding)
            df_test = pd.concat([df_test, df_part], ignore_index=True)

        # Mark train/test splits
        df_train["split"] = "train"
        df_test["split"] = "test"

        # Combine train and test annotations
        df = pd.concat([df_train, df_test], ignore_index=True)

        # Retrieve raw text for each filename from both train and test text folders
        df_texts = handlers.get_texts(train_text_dir, test_text_dir, encoding=self.encoding)
        # Merge annotation DataFrame with text DataFrame on "filename"
        df = df.merge(df_texts, on="filename", how="left")

        # Ensure no duplicate filename+mark combinations
        assert df.duplicated(subset=["filename", "mark"]).sum() == 0, "Found duplicated filename+mark entries."

        # Store internally and return
        self.df = df
        return df

    def preprocess_data(self) -> None:
        """
        Standardize column names and ensure all expected columns exist.

        - Creates "filenameid" as "filename#off0#off1".
        - Drops old columns ["filename", "off0", "off1"].
        - Renames "filename" -> "filenameid", "label" -> "mention_class", "semantic_rel" -> "sem_rel".
        - Adds any missing columns from DS_COLUMNS with None values.
        - Reorders columns so that DS_COLUMNS appear first, followed by ["text", "split"].

        Raises:
            AssertionError: If after processing, some DS_COLUMNS are still missing.
        """
        # Map old names to standardized column names
        rename_map = {
            "filename": "filenameid",
            "label": "mention_class",
            "semantic_rel": "sem_rel"
        }

        # Create "filenameid" by concatenating filename and offsets
        self.df["filenameid"] = self.df["filename"] + "#" + self.df["off0"] + "#" + self.df["off1"]
        # Drop the no-longer-needed columns
        self.df.drop(columns=["filename", "off0", "off1"], inplace=True)

        # Rename annotation columns
        self.df.rename(columns=rename_map, inplace=True)

        # Ensure all expected DS_COLUMNS exist
        for col in self.DS_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = None

        # Reorder columns: DS_COLUMNS first, then text and split
        final_cols = self.DS_COLUMNS + ["text", "split"]
        self.df = self.df.loc[:, final_cols]

        # Verify no missing DS_COLUMNS
        missing = set(self.DS_COLUMNS) - set(self.df.columns)
        assert not missing, f"Missing columns after preprocessing: {missing}"

    def _download_data(self, download_path: str) -> str:
        """
        Download and extract the Distemist ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Distemist dataset...")
        temp_zip_path = os.path.join(download_path, "distemist_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Distemist dataset downloaded and extracted successfully.")

        return download_path


class DistemistGazetteer(BenchmarkDataset):
    """
    Loader for the Distemist gazetteer (dictionary of disease terms).

    Inherits from BenchmarkDataset but skips the usual load step if the file already exists.
    Downloads a TSV file, loads it into a DataFrame, and subsets to the required columns.

    Attributes:
        URL (str): Download URL for the Distemist gazetteer TSV.
        NAME (str): Base name (without extension) for the TSV file.
    """

    URL: str = "https://zenodo.org/records/6505583/files/dictionary_distemist.tsv?download=1"
    NAME: str = "dictionary_distemist"

    def __init__(
        self,
        lang: str = "es",
        path: Optional[str] = None,
        name: str = NAME,
        url: str = URL,
        download_if_missing: bool = True,
        encoding: str = "utf-8"
    ) -> None:
        """
        Initialize the DistemistGazetteer.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the TSV. If None, uses default in BenchmarkDataset.
            name (str, optional): Filename base (default is NAME).
            url (str, optional): URL to download the gazetteer TSV.
            download_if_missing (bool, optional): If True, downloads data if not found locally.
            encoding (str, optional): File encoding for reading the TSV (default "utf-8").
        """
        # We pass load=False so that BenchmarkDataset does not attempt to call load_data() immediately.
        super().__init__(lang, name, path, url, download_if_missing, load=False, encoding=encoding)

        # Define the expected TSV path (with .tsv extension)
        self.path = os.path.join(self.path, f"{self.NAME}.tsv")

        if not os.path.exists(self.path):
            if download_if_missing:
                print(f"Gazetteer TSV not found at '{self.path}'. Downloading...")
                self._download_data(self.path)
            else:
                raise FileNotFoundError(f"Gazetteer file '{self.path}' not found and download_if_missing=False.")

        # Load and preprocess immediately
        self.load_data()
        self.preprocess_data()

    def load_data(self) -> pd.DataFrame:
        """
        Load the Distemist gazetteer TSV into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all columns from the TSV.
        """
        self.df = pd.read_csv(self.path, sep="\t", dtype=str, encoding=self.encoding)
        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Subset the loaded DataFrame to only the columns specified in config.GZ_COLUMNS.

        Returns:
            pd.DataFrame: DataFrame with columns = config.GZ_COLUMNS.
        """
        # Keep only the required columns for the gazetteer
        self.df = self.df.loc[:, config.GZ_COLUMNS]
        return self.df

    def _download_data(self, download_path: str) -> str:
        """
        Download the gazetteer TSV directly and save it to `download_path`.

        Args:
            download_path (str): Full filepath (including .tsv) where to save the downloaded content.

        Returns:
            str: Path to the saved TSV file.
        """
        print("Downloading Distemist gazetteer...")
        response = get(self.URL)
        response.raise_for_status()

        # Ensure directory exists
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

        # Write TSV file
        with open(download_path, "wb") as f:
            f.write(response.content)

        print(f"Gazetteer saved to {download_path}")
        return download_path
