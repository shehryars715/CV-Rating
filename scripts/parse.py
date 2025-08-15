import google.generativeai as genai
from dotenv import load_dotenv
import os
from typing import Optional

load_dotenv()

class CVParser:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def parse_cv(self, raw_text: str) -> Optional[str]:
        """Parse raw CV text into standardized string format"""
        prompt = """Convert this CV into the following exact example format:
        
        "Education: Bachelors of Data Science at Punjab University, Experience: 2 years as AI researcher at Awaaz AI, GPA: 1, Skills: Python, C++, SQL, JAVA"

        Rules:
        1. Include only these 4 sections: Education, Experience, GPA, Skills
        2. For education, use the highest degree only
        3. For experience, use the most recent or relevant position
        4. For skills, if missing, asume based on other sections of CV and for missing gpa (assume 2)
        6. Always maintain this exact format

        CV Content:
        """ + raw_text  # Truncate to avoid token limits

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error parsing CV: {e}")
            return None


if __name__ == "__main__":
    from pdf import PDFTextExtractor
    
    # Example usage
    pdf_path = input("Enter PDF file path: ").strip('"')
    
    # Step 1: Extract text
    extractor = PDFTextExtractor()
    raw_text = extractor.extract_text(pdf_path)
    
    if raw_text:
        # Step 2: Parse with Gemini
        parser = CVParser()
        parsed_cv = parser.parse_cv(raw_text)
        
        if parsed_cv:
            print("\nParsed CV:\n")
            print(parsed_cv)
        else:
            print("Failed to parse CV")
    else:
        print("No text extracted from PDF")