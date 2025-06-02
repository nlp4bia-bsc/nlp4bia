import pymupdf4llm
import markdown
import re
from abc import ABC, abstractmethod
from typing import Optional


class PDFParser(ABC):
    """
    Abstract base class for extracting text from PDF files.

    Attributes:
        pdf_path (str): Path to the PDF file.
        text (Optional[str]): Extracted text content of the PDF.
    """

    def __init__(self, pdf_path: str) -> None:
        """
        Initialize the PDFParser.

        Args:
            pdf_path (str): Path to the PDF file to parse.
        """
        self.pdf_path: str = pdf_path
        self.text: Optional[str] = self.extract_text()

    @abstractmethod
    def extract_text(self) -> Optional[str]:
        """
        Abstract method to extract text from the PDF.

        Returns:
            Optional[str]: The extracted text, or None if extraction fails.
        """
        pass

    def __repr__(self) -> str:
        """
        Return a concise representation of the parser, including the first 10 characters of the extracted text.

        Returns:
            str: String representation.
        """
        snippet = self.text[:10] if self.text is not None else "None"
        return f"{type(self).__name__}(pdf_path={self.pdf_path}, text={snippet}...)"

    def __str__(self) -> str:
        """
        Return the full extracted text.

        Returns:
            str: The extracted text, or an empty string if None.
        """
        return self.text or ""


class PDFParserMuPDF(PDFParser):
    """
    Extract text from a PDF file using MuPDF via pymupdf4llm.

    This implementation converts each page to Markdown, then to HTML,
    and strips HTML tags to return plain text.

    Attributes:
        pdf_path (str): Path to the PDF file.
        text (Optional[str]): Extracted text content of the PDF.
    """

    def extract_text(self) -> Optional[str]:
        """
        Extract text from the PDF using MuPDF.

        Process:
            1. Convert PDF to Markdown via `pymupdf4llm.to_markdown`.
            2. Convert Markdown to HTML via `markdown.markdown`.
            3. Remove HTML tags using a regular expression.
            4. Clean up extra spaces and remove code formatting markers.

        Returns:
            Optional[str]: The cleaned text content, or None if an error occurs.
        """
        try:
            # Convert PDF to Markdown (no progress bar)
            md_text: str = pymupdf4llm.to_markdown(self.pdf_path, show_progress=False)

            # Convert Markdown to HTML
            html_content: str = markdown.markdown(md_text)

            # Strip HTML tags
            plain_text: str = re.sub(r"<[^>]+>", "", html_content).strip()

            # Collapse multiple spaces into a single space
            plain_text = re.sub(r" +", " ", plain_text)

            # Remove Markdown code fences and bold markers
            plain_text = plain_text.replace("```", "")
            plain_text = plain_text.replace("**", "")

            self.text = plain_text
        except Exception as e:
            print(f"Error extracting text from PDF {self.pdf_path}: {e}")
            self.text = None

        return self.text
