import pandas as pd

from nlp4bia.preprocessor.HashExtractor import ParallelHashExtractor
from nlp4bia.preprocessor.CSVConcatenator import ParallelCSVConcatenator


class HashDeduplicator:
    """Apply deduplication based on hashes. If specified, uses a CSV with hashes."""
    
    def __init__(self, files_or_pattern, output_dir, hashes_csv=None, num_processes=None):
        self.files_or_pattern = files_or_pattern
        self.output_dir = output_dir
        self.hashes_csv = hashes_csv
        self.df_hashes = pd.read_csv(hashes_csv) if hashes_csv is not None else None
        self.num_processes = num_processes if num_processes is not None else multiprocessing.cpu_count()
        
    def deduplicate_files(self, file_list):
        """Deduplicate files based on hashes."""
        if self.df_hashes is not None:
            # Use the hashes from the CSV
            df_hashes = self.df_hashes
        else:
            # Calculate the hashes on the fly
            phe = ParallelHashExtractor(self.output_dir, num_processes=self.num_processes)
            hashes = phe.get_batch_hash(file_list, progress_bar=True)
            df_hashes = pd.DataFrame(hashes, columns=phe.CSV_COLUMNS)
        
        # Deduplicate based on hashes
        df_dedup = df_hashes.drop_duplicates(subset=phe.CSV_COLUMNS[1], keep="first")[phe.CSV_COLUMNS[0]].tolist()
        # print(f"Original number of files: {len(df_hashes)}")
        # print(f"Number of deduplicated files: {len(df_dedup)}")
        return df_dedup
    
    def get_deduplicated_files(self, output_csv=None):
        """Get the deduplicated files."""
        
        pcsv = ParallelCSVConcatenator(self.output_dir, num_processes=self.num_processes)
        ls_files = self.deduplicate_files(self.files_or_pattern)
        return pcsv.get_batch_content(ls_files, output_csv=output_csv)
    
    
# hd = HashDeduplicator("nbs/preprocessor/test_data/test*/**/*.txt", "nbs/preprocessor/", num_processes=2)

# hd.get_deduplicated_files("nbs/preprocessor/deduplicated_contents.csv");
# # hd.deduplicate_files(ls_files);