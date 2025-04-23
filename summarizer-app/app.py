import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import sys  # Keep sys import if needed elsewhere, otherwise remove if only used in the check

load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
FAISS_INDEX_PATH = "faiss_index"


# Function to load the vector store
def load_vector_store():
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )  # Added allow_dangerous_deserialization
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        return None


# Function to create the vector store
def create_vector_store(text_chunks):
    try:
        embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-en")
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
                page_text = page.extract_text()
                if page_text:  # Ensure text was extracted
                    text += page_text
        except PdfReadError as e:
            st.error(
                f"Error reading PDF file: {e}. Please check if the file is properly formatted and try again."
            )
            return None
        except Exception as e: # Catch other potential errors during PDF processing
            st.error(f"An unexpected error occurred while processing PDF {pdf.name}: {e}")
            return None
    return text


def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        try:
            # Ensure reading as bytes first, then decode
            content_bytes = txt.read()
            text += content_bytes.decode("utf-8")
        except UnicodeDecodeError:
             st.warning(f"Could not decode file {txt.name} as UTF-8. Trying with 'latin-1' encoding.")
             try:
                 # Reset buffer position and try reading again with a different encoding
                 txt.seek(0)
                 content_bytes = txt.read()
                 text += content_bytes.decode("latin-1")
             except Exception as e:
                 st.error(f"Error reading TXT file {txt.name} even with fallback encoding: {e}")
                 return None
        except Exception as e:
            st.error(f"Error reading TXT file {txt.name}: {e}")
            return None
    return text


def get_vtt_text(vtt_docs):
    text = ""
    for vtt in vtt_docs:
        try:
            # Ensure reading as bytes first, then decode
            content_bytes = vtt.read()
            text += content_bytes.decode("utf-8")
        except UnicodeDecodeError:
             st.warning(f"Could not decode file {vtt.name} as UTF-8. Trying with 'latin-1' encoding.")
             try:
                 # Reset buffer position and try reading again with a different encoding
                 vtt.seek(0)
                 content_bytes = vtt.read()
                 text += content_bytes.decode("latin-1")
             except Exception as e:
                 st.error(f"Error reading VTT file {vtt.name} even with fallback encoding: {e}")
                 return None
        except Exception as e:
            st.error(f"Error reading VTT file {vtt.name}: {e}")
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
    if not query:
        st.warning("Cannot generate response from empty input.")
        return None
    try:
        llm = Ollama(model=OLLAMA_MODEL)
        # Ensure the prompt template correctly includes the context and the query/question
        # The 'query' variable here is actually the full text, let's pass it as context.
        # We need a placeholder for a 'question' even if it's implicitly the summarization task.
        template = prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
            # Pass the raw_text (query) as the 'query' to the chain,
            # Langchain's RetrievalQA uses this to find relevant docs ('context')
            # and also potentially as the 'question' if not explicitly separated.
            # Let's provide a generic question for the summarization task.
        )
        # Provide a generic question suitable for the prompt's task
        generic_question = "Summarize the provided text according to the format."
        response = qa_chain.run(query=generic_question) # Use generic question here
        # The context will be automatically fetched and inserted by the retriever

        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None


def main():
    st.set_page_config(layout="centered")
    st.title("Meeting Summarizer")
    st.header("Upload your meetings to get a summary")

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        # Try loading existing vector store on first run
        with st.spinner("Loading existing knowledge base..."):
             st.session_state.vector_store = load_vector_store()
             if st.session_state.vector_store:
                 st.success("Existing knowledge base loaded.")
             else:
                 st.info("No existing knowledge base found or error loading. Please upload files to create one.")


    file_uploader = st.file_uploader(
        "Upload your files", accept_multiple_files=True, type=["txt", "vtt", "pdf"]
    )

    prompt_selection = st.selectbox(
        "Select a prompt", list(prompt_options.keys()) + ["Custom"]
    )

    # Add a new input field for custom prompt
    custom_prompt_input = (
        st.text_area("Custom Prompt", "", height=200) if prompt_selection == "Custom" else None
    )

    if st.button("Process Uploaded Files"):
        if not file_uploader:
            st.warning("Please upload at least one file.")
            return # Stop processing if no files are uploaded

        with st.spinner("Processing uploaded files..."):
            pdf_docs = [file for file in file_uploader if file.name.endswith(".pdf")]
            txt_docs = [file for file in file_uploader if file.name.endswith(".txt")]
            vtt_docs = [file for file in file_uploader if file.name.endswith(".vtt")]

            raw_text = ""
            pdf_text = get_pdf_text(pdf_docs)
            if pdf_text is not None: # Check for None in case of error
                raw_text += pdf_text
            else:
                 st.error("Failed to process PDF files.")
                 return # Stop if PDF processing failed

            txt_text = get_txt_text(txt_docs)
            if txt_text is not None: # Check for None in case of error
                raw_text += txt_text
            else:
                st.error("Failed to process TXT files.")
                return # Stop if TXT processing failed

            vtt_text = get_vtt_text(vtt_docs)
            if vtt_text is not None: # Check for None in case of error
                raw_text += vtt_text
            else:
                st.error("Failed to process VTT files.")
                return # Stop if VTT processing failed


            if not raw_text.strip():
                st.warning("No text extracted from the uploaded files.")
                return

            text_chunks = get_text_chunks(raw_text)
            if not text_chunks:
                 st.warning("Could not split text into chunks.")
                 return

            # Create and save the vector store from the new text
            # This overwrites the previous index.
            st.session_state.vector_store = create_vector_store(text_chunks)
            if st.session_state.vector_store is None:
                st.error("Failed to create knowledge base from uploaded files.")
                return

            st.success("Files processed and knowledge base updated.")
            # Store the extracted raw text in session state to use for summarization
            st.session_state.raw_text = raw_text


    # Button to generate summary, only active if vector store exists
    if st.session_state.vector_store is not None and "raw_text" in st.session_state:
        if st.button("Generate Summary"):
             with st.spinner("Generating summary..."):
                # Determine the prompt to use
                if prompt_selection == "Custom":
                    if not custom_prompt_input:
                        st.warning("Please enter a custom prompt.")
                        return
                    selected_prompt = custom_prompt_input
                else:
                    selected_prompt = prompt_options[prompt_selection]

                # Use the stored raw_text for the query/context
                summary = get_ollama_response(st.session_state.vector_store, st.session_state.raw_text, selected_prompt)
                if summary:
                    st.subheader("Generated Summary")
                    st.write(summary)
                else:
                    st.error("Failed to generate summary.")
    elif st.session_state.vector_store is None:
         st.info("Upload and process files to enable summary generation.")


# Call main function directly
main()
