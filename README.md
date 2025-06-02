# NLP4BIA Library

[![PyPI version](https://img.shields.io/pypi/v/nlp4bia.svg)](https://pypi.org/project/nlp4bia)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

This repository provides a Python library for loading, processing, and utilizing biomedical datasets curated by the NLP4BIA research group at the Barcelona Supercomputing Center (BSC). The datasets are specifically designed for natural language processing (NLP) tasks in the biomedical domain.


- [NLP4BIA Library](#nlp4bia-library)
  - [Installation](#installation)
  - [Introduction](#introduction)
  - [Available Dataset Loaders](#available-dataset-loaders)
    - [1. **Distemist**](#1-distemist)
    - [2. **Meddoplace**](#2-meddoplace)
    - [3. **Medprocner**](#3-medprocner)
    - [4. **Symptemist**](#4-symptemist)
      - [Dataset Columns](#dataset-columns)
      - [Gazetteer Columns](#gazetteer-columns)
  - [Quick Start Guide](#quick-start-guide)
    - [Example Usage](#example-usage)
      - [Dataset Loaders](#dataset-loaders)
      - [Preprocessor](#preprocessor)
        - [Deduplication](#deduplication)
        - [Document Parser](#document-parser)
      - [Linking](#linking)
  - [Contributing](#contributing)
  - [License](#license)
  - [References](#references)

---

## Installation

```bash
pip install nlp4bia
```

## Introduction

**NLP4BIA** is a Python package for working with curated biomedical NLP datasets in Spanish. Developed by the NLP4BIA research group at the Barcelona Supercomputing Center (BSC), it provides:

- **Dataset Loaders** for public benchmarks like Distemist, Meddoplace, Medprocner, Symptemist.  
- **Preprocessing Utilities** such as deduplication, PDF parsing, and more.  
- **Linking Tools** to perform dense retrieval against medical gazetteers (e.g., SNOMED CT) using SentenceTransformers.  

Whether you’re training new NLP models on Spanish clinical text, sanitizing raw medical documents, or performing terminology linking, NLP4BIA aims to streamline your workflow.

## Available Dataset Loaders

The library currently supports the following dataset loaders, which are part of public benchmarks:

### 1. **Distemist**
   - **Description**: A dataset for disease mentions recognition and normalization in Spanish medical texts.
   - **Zenodo Repository**: [Distemist Zenodo](https://doi.org/10.5281/zenodo.7614764)

### 2. **Meddoplace**
   - **Description**: A dataset for place name recognition in Spanish medical texts.
   - **Zenodo Repository**: [Meddoplace Zenodo](https://doi.org/10.5281/zenodo.8403498)

### 3. **Medprocner**
   - **Description**: A dataset for procedure name recognition in Spanish medical texts.
   - **Zenodo Repository**: [Medprocner Zenodo](https://doi.org/10.5281/zenodo.7817667)

### 4. **Symptemist**
   - **Description**: A dataset for symptom mentions recognition in Spanish medical texts.
   - **Zenodo Repository**: [Symptemist Zenodo](https://doi.org/10.5281/zenodo.10635215)


#### Dataset Columns
| Column Name       | Type/Example                     | Description                                             |
|-------------------|----------------------------------|---------------------------------------------------------|
| `filenameid`      | `"12345_678"`             | Unique ID combining filename and character offsets.     |
| `mention_class`   | `"ENFERMEDAD"`                      | Class of the mention (disease, symptom, procedure, etc.). |
| `span`            | `"diabetes tipo 2"`             | Text span corresponding to the mention.                 |
| `code`            | `"44054006"`                     | Normalized SNOMED CT code for the mention.              |
| `sem_rel`         | `"EXACT"`/`"NARROW"`/`"COMPOSITE"`                       | EXACT: The mention matches perfectly with the associated term; NARROW: it is not exactly the same but a term parent not in the ontology; COMPOSITE: needs more than one code to be defined (e.g. 1243535+13452543) |
| `is_abbreviation` | `True` / `False`                 | Whether the mention is an abbreviation.                 |
| `is_composite`    | `True` / `False`                 | Whether the mention is a composite term.                |
| `needs_context`   | `True` / `False`                 | Whether extra context is required to interpret the span. |
| `extension_esp`   | `"info adicional"`               | Extra fields specific to Spanish texts.                 |

#### Gazetteer Columns

| Column Name   | Type/Example      | Description                                    |
|---------------|-------------------|------------------------------------------------|
| `code`        | `"44054006"`      | SNOMED CT code for the term.                   |
| `language`    | `"es"`            | Language of the term (e.g., "es", "en").       |
| `term`        | `"diabetes"`      | The term itself (string).                      |
| `semantic_tag`| `"disorder"`      | Semantic tag associated with the term.         |
| `mainterm`    | `True` / `False`  | Whether this is a primary (“preferred”) term or a synonym.  |


---

## Quick Start Guide

### Example Usage

#### Dataset Loaders
Here's how to use one of the dataset loaders, such as `DistemistLoader`:

```python
from nlp4bia.datasets.benchmark.distemist import DistemistLoader

# Initialize loader
distemist_loader = DistemistLoader(lang="es", download_if_missing=True)

# Load and preprocess data
dis_df = distemist_loader.df
print(dis_df.head())
```


Dataset folders are automatically downloaded and extracted to the `~/.nlp4bia` directory.

#### Preprocessor

##### Deduplication

```python
from nlp4bia.preprocessor.deduplicator import HashDeduplicator

# Define the list of files to deduplicate
ls_files = ["path/to/file1.txt", "path/to/file2.txt"]

# Instantiate the deduplicator. It deduplicates the files using 8 cores.
hd = HashDeduplicator(ls_files, num_processes=8)

# Deduplicate the files and save the results to a CSV file
hd.get_deduplicated_files("path/tp/deduplicated_contents.csv")
```

##### Document Parser

**PDFS**

```python
from nlp4bia.preprocessor.pdfparser import PDFParserMuPDF

# Define the path to the PDF file
pdf_path = "path/to/file.pdf"

# Instantiate the PDF parser
pdf_parser = PDFParserMuPDF(pdf_path)

# Extract the text from the PDF file
pdf_text = pdf_parser.extract_text()
```

#### Linking

Perform dense retrieval using the `DenseRetriever` class:

```python
from sentence_transformers import SentenceTransformer
from nlp4bia.datasets.benchmark.medprocner import MedprocnerLoader, MedprocnerGazetteer
from nlp4bia.linking.retrievers import DenseRetriever

# Load the dataset and gazetteer
df_proc = MedprocnerLoader().df
gaz_proc = MedprocnerGazetteer().df
gaz_proc = gaz_proc.sort_values(by=["code", "mainterm"], 
                                ascending=[True, False]) # Make sure mainterms are first

# Load the model
model_name = "path/to/model"
st_model = SentenceTransformer(model_name)

# Create the vector database
vector_db = st_model.encode(gaz_proc["term"].tolist()[:100], 
                            show_progress_bar=True, 
                            convert_to_tensor=True, 
                            normalize_embeddings=True)

# Initialize the retriever
biencoder = DenseRetriever(vector_db=vector_db, model=st_model)
biencoder.retrieve_top_k(["reparación de un desprendimiento de la retina"], 
                          gaz_proc.iloc[:100], 
                          k=10, 
                          input_format="text")
```
---

## Contributing

Contributions to expand the dataset loaders or improve existing functionality are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References

If you use this library or its datasets in your research, please cite the corresponding Zenodo repositories or related publications.
