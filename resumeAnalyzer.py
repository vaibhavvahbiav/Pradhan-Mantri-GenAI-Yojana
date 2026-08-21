import streamlit as st


from utils import extract_pdf, create_vector_store

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📄 AI Resume Analyzer")
st.markdown(
    """
    Analyze your resume against a job description using a local LLM.

    Upload your resume, provide the job description, and get insights
    into your skills, ATS compatibility, and interview preparation.
    """
)

st.divider()


# ---------------------------------------------------------
# Input Section
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("📎 Upload Resume")

    resume_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"]
    )

with col2:
    st.subheader("💼 Job Description")

    jd_text = st.text_area(
        "Paste the job description here",
        height=200,
        placeholder="Paste the complete job description..."
    )


# ---------------------------------------------------------
# Analyze Button
# ---------------------------------------------------------

st.divider()

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True
)


if analyze_button:

    # -----------------------------------------------------
    # Validate Input
    # -----------------------------------------------------

    if not resume_file:
        st.warning("Please upload your resume first.")

    elif not jd_text.strip():
        st.warning("Please provide the job description.")

    else:

        with st.spinner("Analyzing your resume..."):

            # -------------------------------------------------
            # Extract Resume Text
            # -------------------------------------------------

            resume_text = extract_pdf(resume_file)

            # Combine resume + JD
            combined_text = (
                resume_text
                + "\n\nJob Description:\n"
                + jd_text
            )

            # -------------------------------------------------
            # Create Vector Store
            # -------------------------------------------------

            vectorstore = create_vector_store(combined_text)

            retriever = vectorstore.as_retriever()

            # -------------------------------------------------
            # Load Local LLM
            # -------------------------------------------------

            llm = Ollama(
                model="gemma2:2b"
            )

            # -------------------------------------------------
            # Prompt
            # -------------------------------------------------

            prompt = ChatPromptTemplate.from_template(
                """
You are an expert technical recruiter and resume reviewer.

Analyze the candidate's resume against the provided job description.

Use only the information available in the context.
Do not invent skills, experience, projects, or qualifications.

Provide a practical and honest assessment.

Context:
{context}

Question:
{question}

Your analysis must include:

1. Skills Gap Analysis
   - Skills the candidate already has
   - Skills required by the job that appear to be missing

2. Resume Score
   - Score from 1 to 10
   - Brief explanation

3. Missing Skills
   - Technical skills
   - Soft skills
   - Domain knowledge

4. Missing Technologies
   - Programming languages
   - Frameworks
   - Tools
   - Hardware/software technologies

5. ATS Compatibility Score
   - Score from 0 to 100
   - Explain important missing keywords

6. Technical Interview Questions
   - Generate 10 questions based specifically on the job description
   - Prioritize questions the candidate is likely to encounter

7. Resume Improvement Suggestions
   - Specific improvements to projects
   - Skills section improvements
   - Experience descriptions
   - Keywords to add
   - Things that should be removed or rewritten

Be honest and constructive.
Prioritize actionable feedback over generic advice.
"""
            )

            # -------------------------------------------------
            # RAG Chain
            # -------------------------------------------------

            chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # -------------------------------------------------
            # Run Analysis
            # -------------------------------------------------

            response = chain.invoke(
                "Analyze this resume against the provided job description."
            )

        # -----------------------------------------------------
        # Display Results
        # -----------------------------------------------------

        st.success("Resume analysis completed!")

        st.subheader("📊 Resume Analysis")

        st.markdown(response)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Powered by LangChain + Ollama + FAISS"
)
