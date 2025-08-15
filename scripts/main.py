import streamlit as st
from io import BytesIO
from typing import Union, Optional
import os
from pdf import PDFTextExtractor  # Assuming you have the correct module
from predict import predict_rating  # Assuming you have the correct module
from parse import CVParser  # Assuming you have the correct module

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
            st.subheader("Summary:")
            st.text(parsed_cv)
            
            # Step 4: Predict the rating based on parsed CV (via predict_rating)
            predicted_rating = predict_rating(parsed_cv)
            st.write(f"Predicted Rating: {predicted_rating}")
        else:
            st.error("Failed to parse CV")
    else:
        st.error("No text extracted from PDF")

def main():
    st.title("CV Processing and Rating Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")
    
    if uploaded_file is not None:
        # Show the uploaded file's name
        st.write("Uploaded file:", uploaded_file.name)
        
        # Process the CV
        process_cv(uploaded_file)

if __name__ == "__main__":
    main()
