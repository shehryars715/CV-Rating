from io import BytesIO
import os
from PyPDF2 import PdfReader
from typing import Optional, Union
import re

class PDFTextExtractor:
    def extract_texts(self, pdf_path: str) -> Optional[str]:
        """Extract raw text from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                text = "\n".join([page.extract_text() for page in reader.pages])
                return text if text.strip() else None
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None
        
    def extract_text(self, pdf_file: Union[bytes, BytesIO, str]) -> Optional[str]:
        """Handle bytes, BytesIO, or file path"""
        try:
            if isinstance(pdf_file, bytes):
                file_obj = BytesIO(pdf_file)  # Convert bytes to file-like object
            elif isinstance(pdf_file, BytesIO):
                file_obj = pdf_file  # Use directly if already BytesIO
        
            reader = PdfReader(file_obj)
            text = "\n".join([page.extract_text() for page in reader.pages])
            return text if text.strip() else None
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove common PDF artifacts
        text = re.sub(r'page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)  # Dates
        text = re.sub(r'[^\w\s.,;:!?@#$%&*+-=]', '', text)  # Special chars
        
        return text

    

if __name__ == "__main__":
    extractor = PDFTextExtractor()
    pdf_path = input("Enter PDF file path: ").strip('"')
    
    if not os.path.exists(pdf_path):
        print("Error: File does not exist")
    else:
        extracted_text = extractor.extract_text(pdf_path)
        if extracted_text:
            print(extracted_text)
        else:
            print("Failed to extract text from PDF")