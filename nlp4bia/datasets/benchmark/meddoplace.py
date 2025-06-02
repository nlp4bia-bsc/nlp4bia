import os
import glob
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from requests import get

from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
from nlp4bia.datasets.utils import handlers


class MeddoplaceLoader(BenchmarkDataset):
    """
    Loader for the Meddoplace dataset (place name recognition in Spanish medical texts).

    Inherits from BenchmarkDataset, which provides download and path management.

    Attributes:
        URL (str): Download URL for the Meddoplace ZIP archive.
        NAME (str): Name of the extracted dataset folder.
        DS_COLUMNS (List[str]): Expected columns for the dataset after preprocessing.
    """

    URL: str = (
        "https://zenodo.org/records/8403498/files/"
        "meddoplace_train+test+gazz+crossmap+multilingual_231003.zip?download=1"
    )
    NAME: str = "meddoplace_train+test+gazz+crossmap+multilingual"
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
        Initialize the MeddoplaceLoader.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the dataset.
                If None, uses default in BenchmarkDataset.
            name (str, optional): Dataset folder name (default is NAME).
            url (str, optional): URL to download the Meddoplace ZIP file.
            download_if_missing (bool, optional): If True, downloads data when not present.
            encoding (str, optional): File encoding for reading TSV/CSV (default "utf-8").
        """
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self) -> pd.DataFrame:
        """
        Load Meddoplace train and test annotations, merge them, and attach raw text.

        Expects the following folder structure under self.path after extraction:
            - meddoplace_train/tsv/meddoplace_tsv_train_complete.tsv
            - meddoplace_train/txt/
            - meddoplace_test/tsv/meddoplace_tsv_test_complete.tsv
            - meddoplace_test/txt/

        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                ["filename", "start_span", "end_span", "label", "span", "code", "semantic_rel", "needs_context", "split", "text"].
        """
        # Paths to annotation TSVs
        train_tsv = os.path.join(self.path, "meddoplace_train", "tsv", "meddoplace_tsv_train_complete.tsv")
        test_tsv = os.path.join(self.path, "meddoplace_test", "tsv", "meddoplace_tsv_test_complete.tsv")

        # Paths to raw text folders
        train_txt_dir = os.path.join(self.path, "meddoplace_train", "txt")
        test_txt_dir = os.path.join(self.path, "meddoplace_test", "txt")

        # Read train and test TSVs
        df_train = pd.read_csv(train_tsv, sep="\t", dtype=str, encoding=self.encoding)
        df_test = pd.read_csv(test_tsv, sep="\t", dtype=str, encoding=self.encoding)

        # Mark splits
        df_train["split"] = "train"
        df_test["split"] = "test"

        # Concatenate train and test
        df = pd.concat([df_train, df_test], ignore_index=True)

        # Rename the column "text" (which actually contains the textual mark/span) to "span"
        if "text" in df.columns:
            df.rename(columns={"text": "span"}, inplace=True)

        # Load raw text files for each filename
        df_texts = handlers.get_texts(train_txt_dir, test_txt_dir, encoding=self.encoding)
        # Merge annotations with raw text on "filename"
        df = df.merge(df_texts, on="filename", how="left")

        # Store internally and return
        self.df = df
        return df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Standardize column names and ensure all expected columns exist.

        - Creates "filenameid" as "filename#start_span#end_span".
        - Drops columns ["filename", "start_span", "end_span"].
        - Renames "label" -> "mention_class" and "need_context" -> "needs_context".
        - Adds any missing columns from DS_COLUMNS with None values.
        - Reorders columns so that DS_COLUMNS appear first, followed by ["text", "split"].

        Returns:
            pd.DataFrame: Preprocessed DataFrame with columns = DS_COLUMNS + ["text", "split"].

        Raises:
            AssertionError: If after processing, some DS_COLUMNS are still missing.
        """
        # Column rename mapping
        rename_map = {"label": "mention_class", "need_context": "needs_context"}
        span_cols = ["start_span", "end_span"]

        # Create "filenameid"
        self.df["filenameid"] = self.df["filename"] + "#" + self.df[span_cols[0]] + "#" + self.df[span_cols[1]]
        # Drop the old columns
        self.df.drop(columns=["filename"] + span_cols, inplace=True)

        # Rename label and need_context
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

        return self.df

    def _download_data(self, download_path: str) -> str:
        """
        Download and extract the Meddoplace ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Meddoplace dataset...")
        temp_zip_path = os.path.join(download_path, "meddoplace_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Meddoplace dataset downloaded and extracted successfully.")

        return download_path


class MeddoplaceGazetteer(BenchmarkDataset):
    """
    Loader for the Meddoplace gazetteer (SNOMED CT place terms).

    Inherits from BenchmarkDataset. Downloads a TSV file containing the gazetteer,
    loads it into a DataFrame, and subsets to the required columns.

    Attributes:
        URL (str): Download URL for the Meddoplace gazetteer ZIP archive.
        NAME (str): Name of the extracted dataset folder (same as loader).
    """

    URL: str = (
        "https://zenodo.org/records/8403498/files/"
        "meddoplace_train+test+gazz+crossmap+multilingual_231003.zip?download=1"
    )
    NAME: str = "meddoplace_train+test+gazz+crossmap+multilingual"

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
        Initialize the MeddoplaceGazetteer.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the dataset.
                If None, uses default in BenchmarkDataset.
            name (str, optional): Dataset folder name (default is NAME).
            url (str, optional): URL to download the gazetteer ZIP file.
            download_if_missing (bool, optional): If True, downloads data when not present.
            encoding (str, optional): File encoding for reading TSV (default "utf-8").
        """
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self) -> pd.DataFrame:
        """
        Load the Meddoplace gazetteer TSV into a DataFrame.

        The gazetteer file is expected at:
            <self.path>/meddoplace_gazetteer/gazetteer_snomed_meddoplace.tsv

        Returns:
            pd.DataFrame: DataFrame containing all columns from the TSV.
        """
        gaz_path = os.path.join(self.path, "meddoplace_gazetteer", "gazetteer_snomed_meddoplace.tsv")
        self.df = pd.read_csv(gaz_path, sep="\t", dtype=str, encoding=self.encoding)
        return self.df

    def preprocess_data(self) -> pd.DataFrame:
        """
        Subset the loaded DataFrame to only the columns specified in config.GZ_COLUMNS.

        Returns:
            pd.DataFrame: DataFrame with columns = config.GZ_COLUMNS.
        """
        self.df = self.df.loc[:, config.GZ_COLUMNS]
        return self.df

    def _download_data(self, download_path: str) -> str:
        """
        Download and extract the Meddoplace gazetteer ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Meddoplace gazetteer...")
        temp_zip_path = os.path.join(download_path, "meddoplace_gazetteer_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Meddoplace gazetteer downloaded and extracted successfully.")

        return download_path
