from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load model
model = joblib.load('data/model/regression_model.pkl')

# Initialize Gemini Embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    task_type="retrieval_document"
)

def predict_rating(cv_text):
    # Get embedding using LangChain
    embedding = embeddings_model.embed_query(cv_text)
    
    # Predict
    rating = model.predict([embedding])[0]
    return round(rating, 1)  # Round to 1 decimal place like your original ratings

# Example usage
if __name__ == "__main__":
    sample_cv = """Education: Bachelors of Data Science at Sargodha University, 
                 Experience: 5 years as AI researcher at Awaaz AI,
                 GPA: 3.5,
                 Skills:  SQL, JAVA, Machine Learning, Deep Learning"""

    predicted_rating = predict_rating(sample_cv)
    print(f"Predicted Rating: {predicted_rating}")