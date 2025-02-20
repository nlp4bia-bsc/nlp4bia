import requests
from tqdm import tqdm
import os
import pandas as pd

def progress_download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
            
def get_texts(*paths, extension=".txt", encoding="utf-8"):
    '''Get texts from text_files
    Input: paths: sequence of paths to text_files
    Output: DataFrame with columns: filename, text
    '''
    ls_texts_path = []
    for path in paths:
        ls_texts_path_i = []
        # For each main path, extract the filenames
        for filename in os.listdir(path):
            if filename.endswith(extension):
                ls_texts_path_i.append((path, filename))
        
        # Append the list of filenames to the main list
        ls_texts_path.extend(ls_texts_path_i)
    
    # Retrieve the text from each file and create tuples with the filename and the content
    ls_texts = [(filename, open(os.path.join(path, filename), encoding=encoding).read()) for (path, filename) in ls_texts_path]
    
    df_texts = pd.DataFrame(ls_texts, columns=["filename", "text"])
    df_texts["filename"] = df_texts["filename"].str.replace(extension, "") # remove the extension
    return df_texts