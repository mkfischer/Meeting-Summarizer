import streamlit as st
from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:27b")
FAISS_INDEX_PATH = "faiss_index"


# Function to load the vector store
def load_vector_store():
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None


# Function to create the vector store
def create_vector_store(text_chunks):
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except PdfReadError as e:
            st.error(
                f"Error reading PDF file: {e}. Please check if the file is properly formatted and try again."
            )
            return None
    return text


def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        try:
            text += txt.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading TXT file: {e}")
            return None
    return text


def get_vtt_text(vtt_docs):
    text = ""
    for vtt in vtt_docs:
        try:
            text += vtt.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading VTT file: {e}")
            return None
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


prompt_options = {
    "Sample Prompt": """
    Generate detailed discussion items of the following meeting transcript do not include minutes of meeting and next steps.
    Do not use any previous information, Use the following format:

    Meeting Overview: <Provide the overview of the meeting>

    Discussion Item 1: <item_name_1>
    - <detailed_description_point_1>
    - <detailed_description_point_2>

    ..

    Discussion Item 2: <item_name_2>
    - <detailed_description_point_1>
    - <detailed_description_point_2>
    ..

    Discussion Item 3: <item_name_3>
    - <detailed_description_point_1>
    - <detailed_description_point_2>

    ..

    Action Item: <item_name_1>
    - <Action_point_1>
    - <Action_point_2>

    ###

    The input text will be appended here: """
}


def get_ollama_response(vector_store, query, prompt):
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt + "{context}",
                    input_variables=["context", "question"],
                )
            },
        )
        response = qa_chain.run(query)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None


def main():
    st.set_page_config(layout="centered")
    st.title("Meeting Summarizer")
    st.header("Upload your meetings to get a summary")

    file_uploader = st.file_uploader(
        "Upload your files", accept_multiple_files=True, type=["txt", "vtt", "pdf"]
    )

    prompt_selection = st.selectbox(
        "Select a prompt", list(prompt_options.keys()) + ["Custom"]
    )

    # Add a new input field for custom prompt
    custom_prompt = (
        st.text_input("Custom Prompt", "") if prompt_selection == "Custom" else None
    )

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            pdf_docs = [file for file in file_uploader if file.name.endswith(".pdf")]
            txt_docs = [file for file in file_uploader if file.name.endswith(".txt")]
            vtt_docs = [file for file in file_uploader if file.name.endswith(".vtt")]

            raw_text = ""
            pdf_text = get_pdf_text(pdf_docs)
            if pdf_text:
                raw_text += pdf_text
            txt_text = get_txt_text(txt_docs)
            if txt_text:
                raw_text += txt_text
            vtt_text = get_vtt_text(vtt_docs)
            if vtt_text:
                raw_text += vtt_text

            if not raw_text:
                st.warning("No text extracted from the uploaded files.")
                return

            text_chunks = get_text_chunks(raw_text)

            # Load vector store if it exists, otherwise create it
            vector_store = load_vector_store()
            if vector_store is None:
                vector_store = create_vector_store(text_chunks)
                if vector_store is None:
                    return

            st.success("Done")

            # Use the custom prompt when "Custom" is selected
            selected_prompt = (
                custom_prompt
                if prompt_selection == "Custom"
                else prompt_options[prompt_selection]
            )

            summary = get_ollama_response(vector_store, raw_text, selected_prompt)
            if summary:
                st.write(summary)


if __name__ == "__main__":
    main()
