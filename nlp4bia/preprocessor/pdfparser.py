import pymupdf4llm, markdown, re
from abc import ABC, abstractmethod


class PDFParser(ABC):
    """
    Abstract class for extracting text from PDF files.
    
    Attributes:
        pdf_path (str): The path to the PDF file.
        text (str): The extracted text.
    """
    
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self.extract_text()
    
    @abstractmethod
    def extract_text(self):
        pass
    
    def __repr__(self):
        return f"{type(self).__name__}(pdf_path={self.pdf_path}, text={self.text[:10]}...)"
    
    def __str__(self):
        return self.text

class PDFParserMuPDF(PDFParser):
    """
    Extract text from PDF files using the pymupdf4llm method from MuPDF library.
    
    Attributes:
        pdf_path (str): The path to the PDF file.
        text (str): The extracted text.
    
    Methods:
        extract_text: Extract text from the PDF file. Returns text (None if an error occurs). 
    """
    def extract_text(self):
        try:
            md_text = pymupdf4llm.to_markdown(self.pdf_path, show_progress=False)
            html_content = markdown.markdown(md_text)
            self.text = re.sub(r'<[^>]+>', '', html_content).strip()
            self.text = re.sub(r' +', ' ', self.text)
            self.text = self.text.replace("```", "")
            self.text = self.text.replace("**", "")
        except Exception as e:
            print(f"Error extracting text from PDF {self.pdf_path}: {e}")
            self.text = None
        return self.text
    