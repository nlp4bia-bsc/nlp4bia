# NLP4BIA Library

This repository provides a Python library for loading, processing, and utilizing biomedical datasets curated by the NLP4BIA research group at the Barcelona Supercomputing Center (BSC). The datasets are specifically designed for natural language processing (NLP) tasks in the biomedical domain.

---

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
- **filenameid**: Unique identifier combining filename and offset information.
- **mention_class**: The class of the mention (e.g., disease, symptom, etc.).
- **span**: Text span corresponding to the mention.
- **code**: The normalized code for the mention (usually to SNOMED CT).
- **sem_rel**: Semantic relationships associated with the mention.
- **is_abbreviation**: Indicates if the mention is an abbreviation.
- **is_composite**: Indicates if the mention is a composite term.
- **needs_context**: Indicates if the mention requires additional context.
- **extension_esp**: Additional information specific to Spanish texts.

#### Gazetteer Columns
- **code**: Normalized code for the term.
- **language**: Language of the term.
- **term**: The term itself.
- **semantic_tag**: Semantic tag associated with the term.
- **mainterm**: Indicates if the term is a primary term.

---

## Installation

```bash
pip install nlp4bia
```

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
vector_db = st_model.encode(gaz_proc["term"].tolist()[:100], show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

# Initialize the retriever
biencoder = DenseRetriever(vector_db=vector_db, model=st_model)
biencoder.retrieve_top_k(["reparación de un desprendimiento de la retina"], gaz_proc.iloc[:100], k=10, input_format="text")
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

---
# Instructions for Maintainers

1. Update the version in `nlp4bia/__init__.py` and in `pyproject.toml`.
2. Remove the `dist` folder (`rm -rf dist`).
3. Build the package (`python -m build`).
4. Check the package (`twine check dist/*`).
5. Upload the package (`twine upload dist/*`).
6. Install the package (`pip install nlp4bia`).

Note: to build you have to install `build` and `twine` packages:
```bash
pip install build twine
```
