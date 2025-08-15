import streamlit as st
from io import BytesIO
import time
from pdf import PDFTextExtractor
from predict import predict_rating
from parse import CVParser

# --- Page Config ---
st.set_page_config(
    page_title="CV Analyzer Pro | AI Engineer (RAG) Job Matcher",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stFileUploader>div>div>div>button {
        background-color: #4285F4;
        color: white;
    }
    .stProgress>div>div>div>div {
        background-color: #4285F4;
    }
    .metric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .jd-box {
        background-color: #e6f7ff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .match-success {
        color: #28a745;
        font-weight: bold;
    }
    .match-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .match-danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Job Description for AI Engineer (RAG Pipeline) ---
AI_ENGINEER_JD = {
    "title": "AI Engineer - RAG Pipeline Specialist",
    "required_skills": [
        "Python", "LangChain", "LLMs (GPT-4, Llama 2)", "Vector Databases (FAISS, Pinecone)",
        "Transformer Models", "Information Retrieval", "NLP", "PyTorch/TensorFlow",
        "API Development (FastAPI, Flask)", "Cloud (AWS, GCP, Azure)"
    ],
    "preferred_skills": [
        "Fine-tuning LLMs", "Docker/Kubernetes", "MLOps", "Semantic Search",
        "Knowledge Graphs", "Evaluation Metrics (NDCG, MRR)"
    ],
    "experience": "3+ years in NLP/ML engineering",
    "education": "Bachelor's/Master's in CS, AI, or related field"
}

# --- Process CV & Match with JD ---
def process_cv(pdf_file):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Extract text
    status_text.text("📄 Extracting text from PDF...")
    extractor = PDFTextExtractor()
    raw_text = extractor.extract_text(pdf_file)
    progress_bar.progress(25)

    if raw_text:
        # Step 2: Clean text
        status_text.text("🧹 Cleaning extracted text...")
        cleaned_text = extractor.clean_text(raw_text)
        progress_bar.progress(50)

        # Step 3: Parse CV
        status_text.text("🔍 Analyzing CV content...")
        parser = CVParser()
        parsed_cv = parser.parse_cv(cleaned_text)
        progress_bar.progress(75)

        if parsed_cv:
            # Step 4: Predict rating & match with JD
            status_text.text("📊 Generating insights & job match...")
            predicted_rating = predict_rating(parsed_cv)
            progress_bar.progress(100)

            # --- Display Results ---
            st.success("✅ Analysis Complete!")
            
            # Layout: Left (CV Summary), Right (Rating & JD Match)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📝 CV Summary")
                st.markdown(f"""
                <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;">
                    {parsed_cv.replace("\n", "<br>")}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("⭐ Overall Rating")
                rating_emoji = "⭐" * int(round(predicted_rating))
                st.markdown(f"""
                <div style="text-align:center;padding:20px;background-color:#fff4e6;border-radius:10px;">
                    <h1 style="font-size:48px;margin:0;">{predicted_rating:.1f}</h1>
                    <p style="font-size:24px;margin:0;">{rating_emoji}</p>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    label="📥 Download Analysis",
                    data=f"CV Rating: {predicted_rating:.1f}\n\nSummary:\n{parsed_cv}",
                    file_name="cv_analysis.txt",
                    mime="text/plain"
                )

            # --- Job Match Analysis ---
            st.subheader("🔍 Job Match for AI Engineer (RAG Pipeline)")
            
            # Display JD
            with st.expander("📋 View Job Description", expanded=True):
                st.markdown(f"""
                <div class="jd-box">
                    <h4>{AI_ENGINEER_JD['title']}</h4>
                    <p><strong>Required Skills:</strong> {", ".join(AI_ENGINEER_JD['required_skills'])}</p>
                    <p><strong>Preferred Skills:</strong> {", ".join(AI_ENGINEER_JD['preferred_skills'])}</p>
                    <p><strong>Experience:</strong> {AI_ENGINEER_JD['experience']}</p>
                    <p><strong>Education:</strong> {AI_ENGINEER_JD['education']}</p>
                </div>
                """, unsafe_allow_html=True)

            # --- Overall Match Score ---
            match_score = round(predicted_rating/10.5*100,1)  # Simulated (replace with actual logic)
            st.subheader(f"📊 Job Match Score: **{match_score}%**")
            st.progress(match_score / 100)

            # --- Improvement Suggestions ---
            with st.expander("💡 How to Improve Your CV for This Role", expanded=True):
                st.markdown("""
                - **Add projects involving RAG pipelines** (e.g., chatbot with document retrieval)
                - **Highlight experience with LangChain or LlamaIndex**
                - **Showcase vector DB knowledge** (FAISS, Pinecone, Weaviate)
                - **Include API development skills** (FastAPI/Flask for deploying models)
                - **Mention cloud experience** (AWS SageMaker, GCP Vertex AI)
                """)

        else:
            progress_bar.empty()
            status_text.empty()
            st.error("Failed to parse CV. Please try with a different file.")
    else:
        progress_bar.empty()
        status_text.empty()
        st.error("No text could be extracted. Is this a scanned PDF?")

# --- Main App ---
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=AI+Recruiter", use_column_width=True)
        st.title("AI Engineer (RAG) CV Analyzer")
        st.markdown("""
        Upload your CV to:
        - Get a **rating** (1-5) 📊
        - Check **job match** for AI Engineer (RAG) 🤖
        - See **missing skills** ❌
        - Receive **improvement tips** 💡
        """)
        
        st.markdown("---")
        st.markdown("### Settings")
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ("Basic", "Detailed", "Technical Deep Dive"),
            index=1
        )
        st.markdown("---")
        st.markdown("⚡ Powered by **LLM & RAG**")

    # Main Content
    st.header("🚀 AI Engineer (RAG Pipeline) CV Analyzer")
    st.markdown("Upload your CV to see how well you match with **RAG-focused AI Engineering roles**.")
    
    # File Upload
    uploaded_file = st.file_uploader(
        "Drag & drop your CV (PDF)",
        type="pdf",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        with st.expander("📄 Uploaded File", expanded=False):
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
        
        if st.button("🔍 Analyze My CV", type="primary"):
            with st.spinner("Analyzing your CV for RAG AI Engineer roles..."):
                process_cv(uploaded_file)

if __name__ == "__main__":
    main()