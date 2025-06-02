import csv
import hashlib
import multiprocessing
import os
import glob
from typing import List, Tuple, Optional, Union

import pandas as pd
from tqdm import tqdm


class HashExtractor:
    """
    Compute SHA-256 hashes for individual files or batches of files.

    Attributes:
        CSV_COLUMNS (List[str]): Column names for the output CSV: ["Filepath", "Hash"].
    """
    CSV_COLUMNS: List[str] = ["Filepath", "Hash"]

    def get_file_hash(self, file_path: str) -> Tuple[str, str]:
        """
        Calculate the SHA-256 hash of a single file.

        Args:
            file_path (str): Path to the file to hash.

        Returns:
            Tuple[str, str]: A tuple (file_path, hex_digest) where:
                - file_path is the original path passed in.
                - hex_digest is the SHA-256 hash string of the file's contents.
        """
        hash_func = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return file_path, hash_func.hexdigest()

    def get_batch_hash(
        self,
        file_list: List[str],
        output_csv: Optional[str] = None,
        progress_bar: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Compute SHA-256 hashes for a list of files, optionally writing results to a CSV.

        Args:
            file_list (List[str]): List of file paths to hash.
            output_csv (str, optional): If provided, write the resulting hashes to this CSV file.
                                        The CSV will have columns ["Filepath", "Hash"].
                                        Defaults to None (no CSV output).
            progress_bar (bool, optional): If True, display a tqdm progress bar over file_list.
                                           Defaults to True.

        Returns:
            List[Tuple[str, str]]: A list of tuples (file_path, hash) for each file in file_list.
        """
        ls_hashes: List[Tuple[str, str]] = []

        iterator = file_list
        if progress_bar:
            iterator = tqdm(file_list, desc=f"Processing {len(file_list)} files", position=0)

        for file_path in iterator:
            ls_hashes.append(self.get_file_hash(file_path))

        if output_csv is not None:
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
                writer.writerows(ls_hashes)

        return ls_hashes


class ParallelHashExtractor:
    """
    Parallelized SHA-256 hash extraction for a large set of files.

    Splits the file list into chunks across multiple processes, computes individual CSV parts,
    and merges them into a single CSV and DataFrame.

    Attributes:
        num_processes (int): Number of worker processes to spawn (defaults to CPU count).
        output_dir (str): Base directory where intermediate and final CSVs will be stored.
        hash_parts_output_dirs (List[str]): List of subdirectories for storing each process's CSV.
        csv_parts_output_dir (str): Directory for collecting CSV parts before merging.
        output_csv (str): Path to the final merged CSV with all hashes.
        CSV_COLUMNS (List[str]): Column names for the final CSV (inherited from HashExtractor).
    """

    def __init__(
        self,
        output_dir: str,
        num_processes: Optional[int] = None
    ) -> None:
        """
        Initialize the ParallelHashExtractor.

        Args:
            output_dir (str): Directory under which to create:
                               - "hash_parts/output_part_i" subfolders for each process i.
                               - "csv_parts" folder for intermediate CSVs.
                               The final merged CSV will be saved as output_dir/hashes.csv.
            num_processes (int, optional): Number of parallel processes to use.
                                           Defaults to multiprocessing.cpu_count().
        """
        self.num_processes: int = num_processes if num_processes is not None else multiprocessing.cpu_count()
        self.output_dir: str = output_dir
        self.hash_parts_output_dirs: List[str] = [
            os.path.join(output_dir, "hash_parts", f"output_part_{i}")
            for i in range(self.num_processes)
        ]
        self.csv_parts_output_dir: str = os.path.join(output_dir, "csv_parts")
        self.output_csv: str = os.path.join(output_dir, "hashes.csv")

        # Ensure directories exist
        os.makedirs(self.csv_parts_output_dir, exist_ok=True)
        for part_dir in self.hash_parts_output_dirs:
            os.makedirs(part_dir, exist_ok=True)

        # Initialize CSV_COLUMNS from HashExtractor
        he = HashExtractor()
        self.CSV_COLUMNS: List[str] = he.CSV_COLUMNS

    def get_batch_hash(
        self,
        files_or_pattern: Union[str, List[str]],
        progress_bar: bool = False,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute hashes for all files matching a glob pattern or a given list, in parallel.

        Each process writes its own "hashes_part_{i}.csv" in its designated subdirectory.
        After all processes finish, this method merges the CSV parts into a single DataFrame
        and saves it to `self.output_csv` (or to `output_csv` if provided).

        Args:
            files_or_pattern (str or List[str]):
                - If str: a glob pattern (e.g., "data/**/*.txt") to search for files recursively.
                - If List[str]: a pre-enumerated list of file paths.
            progress_bar (bool, optional): If True, show a progress bar in each worker’s local hashing.
                                           Defaults to False.
            output_csv (str, optional): Path for the merged CSV. If None, defaults to `self.output_csv`.

        Returns:
            pd.DataFrame: A DataFrame containing all file paths and their SHA-256 hashes,
                          with columns ["Filepath", "Hash"].
        """
        he = HashExtractor()
        final_csv_path = output_csv if output_csv is not None else self.output_csv

        # Resolve file list from pattern or use provided list
        if isinstance(files_or_pattern, str):
            all_files: List[str] = glob.glob(files_or_pattern, recursive=True)
        else:
            all_files = files_or_pattern

        total_files = len(all_files)
        if total_files == 0:
            # Create an empty DataFrame with correct columns if no files found
            empty_df = pd.DataFrame(columns=self.CSV_COLUMNS)
            empty_df.to_csv(final_csv_path, index=False)
            return empty_df

        # Determine chunk sizes for distributing files across processes
        chunk_size = total_files // self.num_processes
        remainder = total_files % self.num_processes

        file_chunks: List[List[str]] = []
        start_idx = 0
        for i in range(self.num_processes):
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            file_chunks.append(all_files[start_idx:end_idx])
            start_idx = end_idx

        # Define per-process output CSV filenames
        part_csv_paths: List[str] = [
            os.path.join(self.hash_parts_output_dirs[i], f"hashes_part_{i}.csv")
            for i in range(self.num_processes)
        ]

        # Prepare a partial function to pass progress_bar and output path
        from functools import partial
        partial_get_hash = partial(
            he.get_batch_hash,
            progress_bar=progress_bar
        )

        # Launch parallel hashing: each process writes its own CSV part
        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(partial_get_hash, zip(file_chunks, part_csv_paths))

        # After all parts are written, merge them into one DataFrame
        part_dfs: List[pd.DataFrame] = []
        for csv_file in part_csv_paths:
            if os.path.exists(csv_file):
                part_dfs.append(pd.read_csv(csv_file))
        if part_dfs:
            merged_df = pd.concat(part_dfs, ignore_index=True)
        else:
            merged_df = pd.DataFrame(columns=self.CSV_COLUMNS)

        # Write out the merged CSV
        merged_df.to_csv(final_csv_path, index=False)

        return merged_df
