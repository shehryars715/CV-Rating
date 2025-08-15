import os
from pdf import PDFTextExtractor
from predict import predict_rating
from parse import CVParser
from io import BytesIO
from typing import Union, Optional

def process_cv(pdf_file: Union[bytes, BytesIO, str]):
    # Step 1: Extract raw text from PDF
    extractor = PDFTextExtractor()
    raw_text = extractor.extract_text(pdf_file)

    if raw_text:
        # Step 2: Clean extracted text
        cleaned_text = extractor.clean_text(raw_text)

        # Step 3: Parse cleaned text with Gemini (via CVParser)
        parser = CVParser()
        parsed_cv = parser.parse_cv(cleaned_text)

        if parsed_cv:
            print("\nSummary:\n")
            print(parsed_cv)
            
            # Step 4: Predict the rating based on parsed CV (via predict_rating)
            predicted_rating = predict_rating(parsed_cv)
            print(f"Predicted Rating: {predicted_rating}")
        else:
            print("Failed to parse CV")
    else:
        print("No text extracted from PDF")

if __name__ == "__main__":
    # Example usage: Prompt the user to enter a PDF file path
    pdf_path = 'scripts/cv.pdf'
    
    if not os.path.exists(pdf_path):
        print("Error: The specified file does not exist.")
    else:
        process_cv(pdf_path)
