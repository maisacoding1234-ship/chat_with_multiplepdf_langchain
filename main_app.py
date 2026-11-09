import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from openai import OpenAI

load_dotenv()

# ===================================================
# Connect to LM Studio
# ===================================================

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    timeout=120.0
)

MODEL_NAME = "gemma-3-1b-it-qat"

# ===================================================
# Extract text from PDFs
# ===================================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text

# ===================================================
# Split text into chunks
# ===================================================

def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)

# ===================================================
# Create FAISS Database
# ===================================================

def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

def ask_llm(context, question):

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        # Check if LM Studio returned an answer
        if not response.choices:
            return "Error: LM Studio returned no answer choices."

        answer = response.choices[0].message.content

        # Check if the answer is empty
        if not answer:
            return "Error: LM Studio returned an empty answer."

        return answer

    except Exception as e:
        return f"LM Studio error: {e}"

# ============================================
# Search FAISS + Ask Model
# ============================================

def user_input(user_question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists("faiss_index/index.faiss"):
        st.warning("Please upload and process PDFs first.")
        return

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(
        user_question,
        k=4
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    answer = ask_llm(context, user_question)

    st.subheader("💬 Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        st.write(context)


# ============================================
# Streamlit UI
# ============================================

def main():

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Chat with Multiple PDF Documents")

    st.write("Upload your PDFs and ask questions about them.")

    # Ask questions
    user_question = st.text_input(
        "Ask a question from the PDF files:"
    )

    if st.button("Ask Question"):
        if user_question.strip():
            user_input(user_question)
        else:
            st.warning("Please enter a question.")

    # Sidebar for uploading PDFs
    with st.sidebar:

        st.header("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.warning("Please upload PDF files.")

            else:

                with st.spinner("Processing..."):

                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error(
                            "No readable text found in the PDFs."
                        )
                        return

                    chunks = get_text_chunks(raw_text)

                    if not chunks:
                        st.error("No text chunks were created.")
                        return

                    get_vector_store(chunks)

                st.success(
                    "FAISS database created successfully!"
                )


# ============================================
# Run Application
# ============================================

if __name__ == "__main__":
    main()
