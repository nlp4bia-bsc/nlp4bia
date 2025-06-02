import os
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from requests import get

from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
from nlp4bia.datasets.utils import handlers


class SymptemistLoader(BenchmarkDataset):
    """
    Loader for the Symptemist dataset (symptom mention recognition in Spanish medical texts).

    Inherits from BenchmarkDataset, which handles download and base path management.

    Attributes:
        URL (str): Download URL for the Symptemist ZIP archive.
        NAME (str): Name of the extracted dataset folder.
        DS_COLUMNS (List[str]): Expected columns for the dataset after preprocessing.
    """

    URL: str = "https://zenodo.org/records/10635215/files/symptemist-complete_240208.zip?download=1"
    NAME: str = "symptemist-complete_240208"
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
        Initialize the SymptemistLoader.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the dataset.
                If None, uses default defined in BenchmarkDataset.
            name (str, optional): Dataset folder name (default is NAME).
            url (str, optional): URL to download the Symptemist ZIP file.
            download_if_missing (bool, optional): If True, downloads data when not present locally.
            encoding (str, optional): File encoding for reading TSV/CSV files (default "utf-8").
        """
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self) -> pd.DataFrame:
        """
        Load Symptemist train and test annotations, merge them, and attach raw text.

        Expects the following folder structure under self.path after extraction:
            - symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete+COMPOSITE.tsv
            - symptemist_train/subtask1-ner/txt/
            - symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2+COMPOSITE.tsv
            - symptemist_test/subtask1-ner/txt/

        The train and test TSVs contain columns:
            ["filename", "span_ini", "span_end", "label", "text", "code", "semantic_rel", "need_context", ...].

        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                ["filename", "span_ini", "span_end", "label", "span", "code", "semantic_rel", "need_context", "split", "text"].
        """
        # Paths to annotation TSV files
        train_tsv: str = os.path.join(
            self.path,
            "symptemist_train",
            "subtask2-linking",
            "symptemist_tsv_train_subtask2_complete+COMPOSITE.tsv"
        )
        test_tsv: str = os.path.join(
            self.path,
            "symptemist_test",
            "subtask2-linking",
            "symptemist_tsv_test_subtask2+COMPOSITE.tsv"
        )

        # Paths to raw text directories
        train_txt_dir: str = os.path.join(self.path, "symptemist_train", "subtask1-ner", "txt")
        test_txt_dir: str = os.path.join(self.path, "symptemist_test", "subtask1-ner", "txt")

        # Read train and test annotation TSVs
        df_train: pd.DataFrame = pd.read_csv(train_tsv, sep="\t", dtype=str, encoding=self.encoding)
        df_test: pd.DataFrame = pd.read_csv(test_tsv, sep="\t", dtype=str, encoding=self.encoding)

        # Mark train/test splits
        df_train["split"] = "train"
        df_test["split"] = "test"

        # Concatenate train and test DataFrames
        df: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)

        # Rename the column "text" (which actually contains raw text for the document) to "span"
        if "text" in df.columns:
            df.rename(columns={"text": "span"}, inplace=True)

        # Load raw text files for each filename using utils.get_texts
        df_texts: pd.DataFrame = handlers.get_texts(train_txt_dir, test_txt_dir, encoding=self.encoding)
        # Merge annotation DataFrame with text DataFrame on "filename"
        df = df.merge(df_texts, on="filename", how="left")

        # Store internally and return
        self.df = df
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Standardize column names and ensure all expected columns exist.

        - Creates "filenameid" as "filename#span_ini#span_end".
        - Drops columns ["filename", "span_ini", "span_end"].
        - Renames "label" -> "mention_class" and "need_context" -> "needs_context".
        - Adds any missing columns from DS_COLUMNS with None values.
        - Reorders columns so that DS_COLUMNS appear first, followed by ["text", "split"].

        Returns:
            pd.DataFrame: Preprocessed DataFrame with columns = DS_COLUMNS + ["text", "split"].

        Raises:
            AssertionError: If after processing, some DS_COLUMNS are still missing.
        """
        # Map original column names to standardized names
        rename_map = {"label": "mention_class", "need_context": "needs_context"}
        span_cols = ["span_ini", "span_end"]

        # Create "filenameid" by concatenating filename and span offsets
        self.df["filenameid"] = (
            self.df["filename"] + "#" + self.df[span_cols[0]] + "#" + self.df[span_cols[1]]
        )
        # Drop the original filename and span offset columns
        self.df.drop(columns=["filename"] + span_cols, inplace=True)

        # Rename annotation columns
        self.df.rename(columns=rename_map, inplace=True)

        # Ensure all expected DS_COLUMNS exist in the DataFrame
        for col in self.DS_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = None

        # Reorder columns: DS_COLUMNS first, then "text" (raw document text) and "split"
        final_cols = self.DS_COLUMNS + ["text", "split"]
        self.df = self.df.loc[:, final_cols]

        # Verify that no DS_COLUMNS are missing
        missing = set(self.DS_COLUMNS) - set(self.df.columns)
        assert not missing, f"Missing columns after preprocessing: {missing}"

        return self.df

    def _download_data(self, download_path: str) -> str:
        """
        Download and extract the Symptemist ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Symptemist dataset...")
        temp_zip_path: str = os.path.join(download_path, "symptemist_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Symptemist dataset downloaded and extracted successfully.")

        return download_path


class SymptemistGazetteer(BenchmarkDataset):
    """
    Loader for the Symptemist gazetteer (SNOMED CT symptom terms in Spanish).

    Inherits from BenchmarkDataset. Downloads a ZIP archive containing the gazetteer TSV,
    loads it into a DataFrame, and subsets to the required columns.
    """

    URL: str = "https://zenodo.org/records/10635215/files/symptemist-complete_240208.zip?download=1"
    NAME: str = "symptemist-complete_240208"

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
        Initialize the SymptemistGazetteer.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the dataset.
                If None, uses default defined in BenchmarkDataset.
            name (str, optional): Dataset folder name (default is NAME).
            url (str, optional): URL to download the gazetteer ZIP file.
            download_if_missing (bool, optional): If True, downloads data when not present.
            encoding (str, optional): File encoding for reading TSV (default "utf-8").
        """
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self) -> pd.DataFrame:
        """
        Load the Symptemist gazetteer TSV into a DataFrame.

        The gazetteer file is expected at:
            <self.path>/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv

        Returns:
            pd.DataFrame: DataFrame containing all columns from the gazetteer TSV.
        """
        gaz_path: str = os.path.join(self.path, "symptemist_gazetteer", "symptemist_gazetter_snomed_ES_v2.tsv")
        self.df = pd.read_csv(gaz_path, sep="\t", dtype=str, encoding=self.encoding)
        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Subset the loaded DataFrame to only the columns specified in config.GZ_COLUMNS.

        Returns:
            pd.DataFrame: DataFrame with columns exactly = config.GZ_COLUMNS.
        """
        self.df = self.df.loc[:, config.GZ_COLUMNS]
        return self.df

    def _download_data(self, download_path: str) -> str:
        """
        Download and extract the Symptemist gazetteer ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Symptemist gazetteer...")
        temp_zip_path: str = os.path.join(download_path, "symptemist_gazetteer_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Symptemist gazetteer downloaded and extracted successfully.")

        return download_path