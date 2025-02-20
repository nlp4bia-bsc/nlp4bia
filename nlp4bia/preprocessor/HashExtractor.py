import csv
import hashlib
from tqdm import tqdm
import multiprocessing
import os
import glob
import pandas as pd


class HashExtractor:
    CSV_COLUMNS = ["Filepath", "Hash"]
    
    def get_file_hash(self, file_path):
        """Calculate the SHA-256 hash of a file."""
        hash_func = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return file_path, hash_func.hexdigest()
    
    def get_batch_hash(self, file_list, output_csv=None, progress_bar=True):
        """Process a list of files, calculate hashes, and save to a CSV."""
        ls_hashes = []

        if progress_bar:
            file_list = tqdm(file_list, desc=f"Processing {len(file_list)} files", position=0)

        for file_path in file_list:
                ls_hashes.append(self.get_file_hash(file_path))
        
        if output_csv is not None:
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
                writer.writerows(ls_hashes)
        
        return ls_hashes
    
    
class ParallelHashExtractor:
    
    def __init__(self, output_dir, num_processes=None):
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()
        self.output_dir = output_dir
        self.hash_parts_output_dirs = [f"{output_dir}/hash_parts/output_part_{i}" for i in range(num_processes)]
        self.csv_parts_output_dir = os.path.join(output_dir, "csv_parts")
        self.output_csv = os.path.join(output_dir, "hashes.csv")
        os.makedirs(self.csv_parts_output_dir, exist_ok=True)
        
        for output_dir in self.hash_parts_output_dirs:
            os.makedirs(output_dir, exist_ok=True)
        
    def get_batch_hash(self, files_or_pattern, progress_bar=False, output_csv=None):
        """Parallelized function to scan files, distribute tasks, and merge results."""

        he = HashExtractor()
        self.CSV_COLUMNS = he.CSV_COLUMNS
        
        output_csv = output_csv if output_csv is not None else self.output_csv
        
        print(f"Scanning files in parallel on {self.num_processes} cores...")

        if isinstance(files_or_pattern, str):
            all_files = glob.glob(files_or_pattern, recursive=True)
        else:
            all_files = files_or_pattern

        total_files = len(all_files)

        # Calculate the chunk size
        chunk_size = total_files // self.num_processes
        remainder = total_files % self.num_processes

        # Create the file chunks
        file_chunks = []
        start_idx = 0

        for i in range(self.num_processes):
            # Distribute the remainder files among the chunks
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            file_chunks.append(all_files[start_idx:end_idx])
            start_idx = end_idx
            
        # print(file_chunks)
        # Define output CSVs for each process
        output_csvs = [os.path.join(self.hash_parts_output_dirs[i], f"hashes_part_{i}.csv") for i in range(self.num_processes)]
        
        # Run processes in parallel
        from functools import partial
        partial_get_batch_hash = partial(he.get_batch_hash, progress_bar=progress_bar)
        
        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(partial_get_batch_hash, zip(file_chunks, output_csvs))

        print("All individual CSVs generated. Merging into final CSV...")

        # Merge all CSVs into one
        merged_df = pd.concat([pd.read_csv(csv_file) for csv_file in output_csvs], ignore_index=True)
        merged_df.to_csv(self.output_csv, index=False)

        print(f"Final merged CSV saved as: {self.output_csv}")
        
        return merged_df