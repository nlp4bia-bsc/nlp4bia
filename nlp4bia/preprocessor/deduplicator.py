import pandas as pd
import multiprocessing
from typing import List, Optional, Union

from nlp4bia.preprocessor.HashExtractor import ParallelHashExtractor
from nlp4bia.preprocessor.CSVConcatenator import ParallelCSVConcatenator


class HashDeduplicator:
    """
    Deduplicate a set of text files by computing or loading their hashes in parallel.

    If a CSV of precomputed hashes is provided, uses that directly; otherwise computes hashes on the fly.

    Attributes:
        files_or_pattern (Union[str, List[str]]): Glob pattern or list of file paths to deduplicate.
        output_dir (str): Directory where intermediate files or outputs are stored.
        hashes_csv (Optional[str]): Path to CSV containing precomputed hashes, or None.
        df_hashes (Optional[pd.DataFrame]): Loaded DataFrame of hashes if `hashes_csv` is provided.
        num_processes (int): Number of parallel processes to use (defaults to CPU count).
        progress_bar (bool): Whether to display a progress bar during hashing/concatenation.
    """

    def __init__(
        self,
        files_or_pattern: Union[str, List[str]],
        output_dir: str,
        hashes_csv: Optional[str] = None,
        num_processes: Optional[int] = None,
        progress_bar: bool = False
    ) -> None:
        """
        Initialize the HashDeduplicator.

        Args:
            files_or_pattern (str or List[str]):
                Either a glob pattern (e.g., "data/*.txt") or a list of file paths to deduplicate.
            output_dir (str):
                Directory where intermediate files (e.g., temporary CSVs) will be stored.
            hashes_csv (str, optional):
                Path to a CSV file containing two columns [filename, hash]. If provided, skips hash computation.
            num_processes (int, optional):
                Number of parallel processes to use for hashing or CSV concatenation.
                Defaults to multiprocessing.cpu_count().
            progress_bar (bool, optional):
                If True, show a progress bar during parallel operations.
                Defaults to False.
        """
        self.files_or_pattern: Union[str, List[str]] = files_or_pattern
        self.output_dir: str = output_dir
        self.hashes_csv: Optional[str] = hashes_csv
        self.df_hashes: Optional[pd.DataFrame] = (
            pd.read_csv(hashes_csv) if hashes_csv is not None else None
        )
        self.num_processes: int = (
            num_processes if num_processes is not None else multiprocessing.cpu_count()
        )
        self.progress_bar: bool = progress_bar

    def deduplicate_files(self, file_list: List[str]) -> List[str]:
        """
        Deduplicate a list of file paths based on their hash values.

        If `hashes_csv` was provided during initialization, uses that DataFrame to drop duplicates.
        Otherwise, computes hashes in parallel using ParallelHashExtractor.

        Args:
            file_list (List[str]): List of file paths to deduplicate.

        Returns:
            List[str]: A list of deduplicated file paths (keeps the first occurrence of each unique hash).
        """
        if self.df_hashes is not None:
            # Use precomputed hashes from the provided CSV
            df_hashes = self.df_hashes
            phe = None  # Not needed when CSV is provided
        else:
            # Compute hashes on the fly in parallel
            phe = ParallelHashExtractor(self.output_dir, num_processes=self.num_processes)
            hashes = phe.get_batch_hash(file_list, progress_bar=self.progress_bar)
            df_hashes = pd.DataFrame(hashes, columns=phe.CSV_COLUMNS)

        # Drop duplicates based on the "hash" column (phe.CSV_COLUMNS[1]). Keep the first occurrence.
        # Then extract the "filename" column (phe.CSV_COLUMNS[0]) as a deduplicated list.
        dedup_column = phe.CSV_COLUMNS[1] if phe is not None else df_hashes.columns[1]
        filename_column = phe.CSV_COLUMNS[0] if phe is not None else df_hashes.columns[0]

        df_dedup = df_hashes.drop_duplicates(subset=dedup_column, keep="first")
        deduplicated_files = df_dedup[filename_column].tolist()
        return deduplicated_files

    def get_deduplicated_files(self, output_csv: Optional[str] = None) -> List[str]:
        """
        Retrieve deduplicated file contents, optionally writing them to a CSV.

        This method:
          1. Reads or computes hashes for all files in `self.files_or_pattern`.
          2. Drops duplicates, retaining only the first occurrence per unique hash.
          3. Reads the contents of each deduplicated file in parallel (using CSVConcatenator).
          4. If `output_csv` is provided, writes a single concatenated CSV of file contents to that path.

        Args:
            output_csv (str, optional):
                If provided, the concatenated contents of all deduplicated files
                will be written to this CSV file. If None, contents are returned in-memory.

        Returns:
            List[str]:
                If `output_csv` is None, returns a list of strings, each string being
                the content of one deduplicated file. If `output_csv` is provided,
                still returns the list of strings, but also writes them to `output_csv`.
        """
        # Instantiate the CSV concatenator (to read file contents in parallel)
        pcsv = ParallelCSVConcatenator(self.output_dir, num_processes=self.num_processes)

        # Step 1: Determine which files to deduplicate
        if isinstance(self.files_or_pattern, str):
            # If a glob pattern, resolve matching files (implementation of resolution is inside ParallelCSVConcatenator)
            ls_files = pcsv.resolve_pattern(self.files_or_pattern)
        else:
            ls_files = self.files_or_pattern

        # Step 2: Deduplicate the list of files
        deduped_files = self.deduplicate_files(ls_files)

        # Step 3: Read contents of deduplicated files in parallel, optionally writing to CSV
        contents: List[str] = pcsv.get_batch_content(
            deduped_files,
            output_csv=output_csv,
            progress_bar=self.progress_bar
        )
        return contents
