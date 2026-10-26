import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document

# --- Page Configuration & Layout ---
st.set_page_config(
    page_title="AI Resume Screener Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 AI Resume Screening & Analytics Dashboard")
st.markdown("Automate candidate matching using **TF-IDF, Cosine Similarity**, and interactive **Plotly** analytics visualizations.")

# --- Document Text Extraction Helpers ---
def extract_pdf_text(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_docx_text(uploaded_file):
    try:
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"

# --- Sidebar: Job Description & Config ---
with st.sidebar:
    st.header("🎯 Target Job Description")
    job_title = st.text_input("Role Title", "Senior Python & AI Engineer")
    job_description = st.text_area(
        "Requirements & Skills:",
        "Looking for a Python Developer with experience in Machine Learning, Pandas, NumPy, Streamlit, Plotly, SQL, APIs, Docker, and Natural Language Processing (NLP). Must have 3+ years experience building software apps.",
        height=220
    )
    st.markdown("---")
    st.info("💡 **Instructions:** Upload one or multiple candidate CVs/Resumes in PDF or DOCX format to instantly rank and analyze them.")

# --- Main Interface: Resume Upload ---
st.subheader("📁 Upload Candidate Resumes")
uploaded_files = st.file_uploader(
    "Upload resumes (PDF or Word Documents accepted)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

if uploaded_files and job_description:
    candidates = []

    # Process all uploaded files
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            content = extract_pdf_text(file)
        elif ext == "docx":
            content = extract_docx_text(file)
        else:
            content = ""

        candidates.append({
            "filename": file.name,
            "text": content
        })

    df = pd.DataFrame(candidates)

    # --- AI Matching Computation (TF-IDF & Cosine Similarity) ---
    documents = [job_description] + df["text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Compute similarity between Job Description (index 0) and candidate resumes
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    df["Match Score (%)"] = (similarity_scores * 100).round(2)
    
    # Categorize match ratings
    def assign_category(score):
        if score >= 35:
            return "🔥 Strong Match"
        elif score >= 20:
            return "👍 Moderate Match"
        elif score >= 8:
            return "⚠️ Low Match"
        else:
            return "❌ Poor Match"

    df["Category"] = df["Match Score (%)"].apply(assign_category)
    df = df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)

    # --- Dashboard Metric Cards ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Resumes Scanned", len(df))
    c2.metric("Highest Match Score", f"{df['Match Score (%)'].max():.1f}%" if not df.empty else "0%")
    c3.metric("Average Match Score", f"{df['Match Score (%)'].mean():.1f}%" if not df.empty else "0%")
    c4.metric("Target Role", job_title)
    st.markdown("---")

    # --- Ranked Leaderboard Table ---
    st.subheader("🏆 Candidate Leaderboard")
    st.dataframe(
        df[["filename", "Match Score (%)", "Category"]],
        use_container_width=True
    )

    # --- Plotly Interactive Analytics Charts ---
    st.markdown("---")
    st.subheader("📈 Recruitment & Candidate Analytics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Candidate Match Scores Comparison**")
        fig_bar = px.bar(
            df, 
            x="filename", 
            y="Match Score (%)", 
            color="Match Score (%)",
            color_continuous_scale="Viridis",
            text="Match Score (%)"
        )
        fig_bar.update_layout(xaxis_title="Candidate Resume", yaxis_title="Match Score (%)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with chart_col2:
        st.markdown("**Candidate Fit Category Breakdown**")
        category_counts = df["Category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        
        fig_pie = px.pie(
            category_counts, 
            names="Category", 
            values="Count", 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Individual Candidate Text Inspector ---
    st.markdown("---")
    st.subheader("🔍 Individual Resume Content Inspector")
    selected_resume = st.selectbox("Select candidate file to review extracted text:", df["filename"].tolist())
    
    if selected_resume:
        selected_row = df[df["filename"] == selected_resume].iloc[0]
        st.write(f"**File:** {selected_row['filename']} | **Match Score:** {selected_row['Match Score (%)']}% | **Status:** {selected_row['Category']}")
        with st.expander("View Full Extracted Raw Text"):
            st.text_area("Resume Text", selected_row["text"], height=250, disabled=True)

else:
    st.warning("⚠️ Please provide a job description and upload at least one candidate resume (PDF or DOCX) in the sidebar/main panel to begin screening.")
