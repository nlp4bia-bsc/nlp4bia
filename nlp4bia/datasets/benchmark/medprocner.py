import os
from typing import Optional
from zipfile import ZipFile

import pandas as pd
from requests import get

from nlp4bia.datasets.Dataset import BenchmarkDataset
from nlp4bia.datasets import config
from nlp4bia.datasets.utils import handlers


class MedprocnerLoader(BenchmarkDataset):
    """
    Loader for the Medprocner dataset (procedure name recognition in Spanish medical texts).

    Inherits from BenchmarkDataset, which handles download and path management.

    Attributes:
        URL (str): Download URL for the Medprocner ZIP archive.
        NAME (str): Name of the extracted dataset folder.
        DS_COLUMNS (List[str]): Expected columns for the dataset after preprocessing.
    """

    URL: str = (
        "https://zenodo.org/records/8224056/files/"
        "medprocner_gs_train+test+gazz+multilingual+crossmap_230808.zip?download=1"
    )
    NAME: str = "medprocner_gs_train+test+gazz+multilingual+crossmap_230808"
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
        Initialize the MedprocnerLoader.

        Args:
            lang (str, optional): Language code (default "es").
            path (str, optional): Base directory to store/download the dataset.
                If None, uses default in BenchmarkDataset.
            name (str, optional): Dataset folder name (default is NAME).
            url (str, optional): URL to download the Medprocner ZIP file.
            download_if_missing (bool, optional): If True, downloads data when not present.
            encoding (str, optional): File encoding for reading TSV/CSV (default "utf-8").
        """
        super().__init__(lang, name, path, url, download_if_missing, encoding=encoding)

    def load_data(self) -> pd.DataFrame:
        """
        Load Medprocner train and test annotations, merge them, and attach raw text.

        Expects the following folder structure under self.path after extraction:
            - medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv
            - medprocner_train/txt/
            - medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv
            - medprocner_test/txt/

        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                ["filename", "start_span", "end_span", "label", "span", "code", "semantic_rel", "need_context", "split", "text"].
        """
        # Paths to annotation TSVs
        train_tsv: str = os.path.join(self.path, "medprocner_train", "tsv", "medprocner_tsv_train_subtask2.tsv")
        test_tsv: str = os.path.join(self.path, "medprocner_test", "tsv", "medprocner_tsv_test_subtask2.tsv")

        # Paths to raw text directories
        train_txt_dir: str = os.path.join(self.path, "medprocner_train", "txt")
        test_txt_dir: str = os.path.join(self.path, "medprocner_test", "txt")

        # Read train and test annotation TSVs
        df_train: pd.DataFrame = pd.read_csv(train_tsv, sep="\t", dtype=str, encoding=self.encoding)
        df_test: pd.DataFrame = pd.read_csv(test_tsv, sep="\t", dtype=str, encoding=self.encoding)

        # Mark splits
        df_train["split"] = "train"
        df_test["split"] = "test"

        # Concatenate train and test DataFrames
        df: pd.DataFrame = pd.concat([df_train, df_test], ignore_index=True)

        # Rename the column "text" (which contains the annotation span) to "span"
        if "text" in df.columns:
            df.rename(columns={"text": "span"}, inplace=True)

        # Load raw text files for each filename
        df_texts: pd.DataFrame = handlers.get_texts(train_txt_dir, test_txt_dir, encoding=self.encoding)
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
        rename_map = {"label": "mention_class", "need_context": "needs_context"}
        span_cols = ["start_span", "end_span"]

        # Create "filenameid"
        self.df["filenameid"] = (
            self.df["filename"] + "#" + self.df[span_cols[0]] + "#" + self.df[span_cols[1]]
        )
        # Drop the old columns
        self.df.drop(columns=["filename"] + span_cols, inplace=True)

        # Rename label and need_context columns
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
        Download and extract the Medprocner ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Medprocner dataset...")
        temp_zip_path: str = os.path.join(download_path, "medprocner_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Medprocner dataset downloaded and extracted successfully.")

        return download_path


class MedprocnerGazetteer(BenchmarkDataset):
    """
    Loader for the Medprocner gazetteer (SNOMED CT procedure terms).

    Inherits from BenchmarkDataset. Downloads a ZIP archive containing the gazetteer TSV,
    loads it into a DataFrame, and subsets to the required columns.
    """

    URL: str = (
        "https://zenodo.org/records/8224056/files/"
        "medprocner_gs_train+test+gazz+multilingual+crossmap_230808.zip?download=1"
    )
    NAME: str = "medprocner_gs_train+test+gazz+multilingual+crossmap_230808"

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
        Initialize the MedprocnerGazetteer.

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
        Load the Medprocner gazetteer TSV into a DataFrame.

        The gazetteer file is expected at:
            <self.path>/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv

        Returns:
            pd.DataFrame: DataFrame containing all columns from the TSV.
        """
        gaz_path: str = os.path.join(self.path, "medprocner_gazetteer", "gazzeteer_medprocner_v1_noambiguity.tsv")
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
        Download and extract the Medprocner gazetteer ZIP dataset.

        Args:
            download_path (str): Directory where the ZIP will be saved and extracted.

        Returns:
            str: Path to the extraction directory (same as download_path).
        """
        os.makedirs(download_path, exist_ok=True)

        print("Downloading Medprocner gazetteer...")
        temp_zip_path: str = os.path.join(download_path, "medprocner_gazetteer_temp.zip")
        handlers.progress_download(self.URL, temp_zip_path)

        # Extract ZIP contents into download_path
        with ZipFile(temp_zip_path, "r") as zip_file:
            zip_file.extractall(download_path)

        # Remove temporary ZIP file
        os.remove(temp_zip_path)
        print("Medprocner gazetteer downloaded and extracted successfully.")

        return download_path
