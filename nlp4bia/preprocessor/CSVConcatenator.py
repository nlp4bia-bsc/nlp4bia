from typing import List
import os
import csv
import multiprocessing
import pandas as pd
import glob
from tqdm import tqdm


class CSVConcatenator:
    """Class to read contents of files and save to a CSV."""
    
    CSV_COLUMNS = ["filepath", "text"]
    
    def __init__(self, output_dir: str, num_processes: int=None):
        """Initialize the CSVConcatenator object.
        Inputs:
        - output_dir: Directory to save the CSV files.
        - num_processes: Number of processes to use for parallel processing.
        """
        
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()
        self.output_dir = output_dir
        self.csv_parts_output_dir = os.path.join(output_dir, "csv_parts")
        os.makedirs(self.csv_parts_output_dir, exist_ok=True)
        
 
    def get_file_content(self, file_path: str):
        """Reads a file and returns its name and content (single line).
        Inputs:
        - file_path: Path to the file to read.
        Outputs:
        - file_path: Path to the file.
        - content: Content of the file as a single line.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().replace("\n", " ")
            return file_path, content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return file_path, ""
        
    def get_batch_content(self, file_list: List[str], output_csv: str=None, progress_bar: bool=True):
        """Process a list of files, read contents, and save to a CSV.
        Inputs:
        - file_list: List of file paths to read.
        - output_csv: Path to save the CSV file.
        - progress_bar: Whether to show a progress bar.
        Outputs:
        - ls_contents: List of tuples containing file path and content.
        """
        
        ls_contents = []

        if progress_bar:
            file_list = tqdm(file_list, desc=f"Processing {len(file_list)} files", position=0)

        for file_path in file_list:
            ls_contents.append(self.get_file_content(file_path))
        
        if output_csv is not None:
            with open(output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)
                writer.writerows(ls_contents)
        
        return ls_contents
        


# csvc = CSVConcatenator("nbs/preprocessor/", num_processes=2)
# print(csvc.get_file_content("nbs/preprocessor/test_data/test1/test11/example1.txt"))    
# print(csvc.get_batch_content(ls_files, os.path.join(TEST_PATH, "contents.csv")))

class ParallelCSVConcatenator:
    """Class to parallelize the CSVConcatenator class."""
    
    def __init__(self, output_dir: str, num_processes: str=None):
        """Initialize the ParallelCSVConcatenator object.
        Inputs:
        - output_dir: Directory to save the CSV files.
        - num_processes: Number of processes to use for parallel processing.
        """
        
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()
        self.output_dir = output_dir
        self.csv_parts_output_dir = os.path.join(output_dir, "csv_parts")
        os.makedirs(self.csv_parts_output_dir, exist_ok=True)
    
    def get_batch_content(self, files_or_pattern, progress_bar=False, output_csv=None):
        """Parallelized function to scan files, distribute tasks, and merge results.
        
        Inputs:
        - files_or_pattern: List of file paths or glob pattern to scan.
        - progress_bar: Whether to show a progress bar.
        - output_csv: Path to save the final CSV file.
        Outputs:
        - merged_df: Merged DataFrame containing all the contents.
        """
        
        csvc = CSVConcatenator(self.output_dir, num_processes=self.num_processes)
        self.CSV_COLUMNS = csvc.CSV_COLUMNS
        
        output_csv = output_csv if output_csv is not None else os.path.join(self.output_dir, "contents.csv")
        
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
            
        print(file_chunks)
        # Define output CSVs for each process
        output_csvs = [os.path.join(self.csv_parts_output_dir, f"contents_part_{i}.csv") for i in range(self.num_processes)]
        
        # Run processes in parallel
        from functools import partial
        partial_get_batch_content = partial(csvc.get_batch_content, progress_bar=progress_bar)
        
        with multiprocessing.Pool(self.num_processes) as pool:
            pool.starmap(partial_get_batch_content, zip(file_chunks, output_csvs))

        print("All individual CSVs generated. Merging into final CSV...")

        # Merge all CSVs into one
        merged_df = pd.concat([pd.read_csv(csv_file) for csv_file in output_csvs], ignore_index=True)
        merged_df.to_csv(output_csv, index=False)

        print(f"Final merged CSV saved as: {output_csv}")
        return merged_df
        
        
# pcsv = ParallelCSVConcatenator("nbs/preprocessor/", num_processes=2)
# pcsv.get_batch_content("nbs/preprocessor/test_data/test*/**/*.txt")