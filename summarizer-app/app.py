import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS

# Removed: from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Removed: from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM  # Added new import
from langchain.chains import RetrievalQA
import sys

load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwq:32b-preview-q4_K_M")
FAISS_INDEX_PATH = "faiss_index"


# Function to load the vector store
def load_vector_store():
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    pkl_file = os.path.join(FAISS_INDEX_PATH, "index.pkl")

    # Check if the index files exist before attempting to load
    if (
        not os.path.exists(FAISS_INDEX_PATH)
        or not os.path.exists(index_file)
        or not os.path.exists(pkl_file)
    ):
        # Index doesn't exist, return None without error.
        # The info message will be handled in the main function.
        return None

    try:
        # Use the same embeddings model used for creation if possible, or a default
        # Assuming jina was used for creation based on create_vector_store
        embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-en")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        # Only show error if loading fails when files *do* exist
        st.error(f"Error loading existing vector store: {e}")
        return None


# Function to create the vector store
def create_vector_store(text_chunks):
    try:
        # Ensure the index directory exists
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        # Specify the embeddings model explicitly
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
                f"Error reading PDF file: {pdf.name} - {e}. Please check if the file is properly formatted and try again."
            )
            return None
        except Exception as e:  # Catch other potential errors during PDF processing
            st.error(
                f"An unexpected error occurred while processing PDF {pdf.name}: {e}"
            )
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
            st.warning(
                f"Could not decode file {txt.name} as UTF-8. Trying with 'latin-1' encoding."
            )
            try:
                # Reset buffer position and try reading again with a different encoding
                txt.seek(0)
                content_bytes = txt.read()
                text += content_bytes.decode("latin-1")
            except Exception as e:
                st.error(
                    f"Error reading TXT file {txt.name} even with fallback encoding: {e}"
                )
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
            st.warning(
                f"Could not decode file {vtt.name} as UTF-8. Trying with 'latin-1' encoding."
            )
            try:
                # Reset buffer position and try reading again with a different encoding
                vtt.seek(0)
                content_bytes = vtt.read()
                text += content_bytes.decode("latin-1")
            except Exception as e:
                st.error(
                    f"Error reading VTT file {vtt.name} even with fallback encoding: {e}"
                )
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
    "Defined Prompt": """
    Generate detailed and verbose discussion items of the following meeting transcript do not include minutes of meeting and next steps.
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

    Discussion Item 4: <item_name_4>
    - <detailed_description_point_1>
    - <detailed_description_point_2>

    ..

    Action Item: <item_name_1>
    - <Action_point_1>
    - <Action_point_2>

    ###

    The input text will be appended here: """
}


def get_ollama_response(
    vector_store, query_text, prompt
):  # Renamed query -> query_text for clarity
    if not query_text:
        st.warning("Cannot generate response from empty input text.")
        return None
    try:
        # Use the updated OllamaLLM class
        llm = OllamaLLM(model=OLLAMA_MODEL)

        # The prompt template remains the same, expecting context and question
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
            return_source_documents=False,  # Don't need source docs for summarization
        )

        # Provide a generic question suitable for the prompt's task
        generic_question = "Summarize the provided text according to the format."

        # Use invoke instead of run, passing the query in a dictionary
        # The 'query' key is standard for RetrievalQA chains
        response_dict = qa_chain.invoke({"query": generic_question})

        # Extract the actual result from the response dictionary
        # The result is typically under the 'result' key for RetrievalQA
        response = response_dict.get("result")

        if not response:
            st.error("Received an empty response from the language model.")
            return None

        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        # Optionally log the full traceback here for debugging
        # import traceback
        # st.error(traceback.format_exc())
        return None


def main():
    st.set_page_config(layout="centered")
    st.title("Meeting Summarizer")
    st.header("Upload your meetings to get a summary")

    # Initialize session state for vector store and raw text
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None

    # Load vector store only once at the start if not already loaded
    if st.session_state.vector_store is None:
        with st.spinner("Checking for existing knowledge base..."):
            st.session_state.vector_store = load_vector_store()
            if st.session_state.vector_store:
                st.success("Existing knowledge base loaded.")
                # If we load an existing store, we don't have the raw text in session
                # We could potentially store/load raw text too, but for now,
                # require reprocessing if using an old index.
                st.info(
                    "Existing knowledge base loaded. Process files again to generate a new summary."
                )
                st.session_state.raw_text = (
                    None  # Ensure raw_text is cleared if only loading index
                )
            else:
                st.info(
                    "No existing knowledge base found. Please upload files to create one."
                )

    file_uploader = st.file_uploader(
        "Upload your files", accept_multiple_files=True, type=["txt", "vtt", "pdf"]
    )

    prompt_selection = st.selectbox(
        "Select a prompt", list(prompt_options.keys()) + ["Custom"]
    )

    # Add a new input field for custom prompt
    custom_prompt_input = (
        st.text_area("Custom Prompt", "", height=200)
        if prompt_selection == "Custom"
        else None
    )

    if st.button("Process Uploaded Files"):
        if not file_uploader:
            st.warning("Please upload at least one file.")
            st.stop()  # Use st.stop() to halt execution cleanly

        with st.spinner("Processing uploaded files..."):
            pdf_docs = [file for file in file_uploader if file.name.endswith(".pdf")]
            txt_docs = [file for file in file_uploader if file.name.endswith(".txt")]
            vtt_docs = [file for file in file_uploader if file.name.endswith(".vtt")]

            # Reset raw text before processing new files
            st.session_state.raw_text = ""
            processing_error = False  # Flag to track errors

            pdf_text = get_pdf_text(pdf_docs)
            if pdf_text is not None:
                st.session_state.raw_text += pdf_text
            elif pdf_docs:  # Only set error flag if there were PDFs to process
                st.error("Failed to process one or more PDF files.")
                processing_error = True

            txt_text = get_txt_text(txt_docs)
            if txt_text is not None:
                st.session_state.raw_text += txt_text
            elif txt_docs:  # Only set error flag if there were TXTs to process
                st.error("Failed to process one or more TXT files.")
                processing_error = True

            vtt_text = get_vtt_text(vtt_docs)
            if vtt_text is not None:
                st.session_state.raw_text += vtt_text
            elif vtt_docs:  # Only set error flag if there were VTTs to process
                st.error("Failed to process one or more VTT files.")
                processing_error = True

            # If any processing failed, stop
            if processing_error:
                st.session_state.raw_text = None  # Clear partial text
                st.stop()

            if not st.session_state.raw_text or not st.session_state.raw_text.strip():
                st.warning("No text extracted from the uploaded files.")
                st.session_state.raw_text = None  # Clear empty text
                st.stop()

            text_chunks = get_text_chunks(st.session_state.raw_text)
            if not text_chunks:
                st.warning("Could not split text into chunks.")
                st.session_state.raw_text = None  # Clear text if chunking failed
                st.stop()

            # Create and save the vector store from the new text
            # This overwrites the previous index.
            new_vector_store = create_vector_store(text_chunks)
            if new_vector_store is None:
                st.error("Failed to create knowledge base from uploaded files.")
                st.session_state.raw_text = None  # Clear text if store creation failed
                st.stop()
            else:
                st.session_state.vector_store = new_vector_store
                st.success("Files processed and knowledge base updated.")
                # Raw text is already in st.session_state.raw_text

    # Button to generate summary, enabled only if vector store and raw_text exist
    generate_enabled = (
        st.session_state.vector_store is not None
        and st.session_state.raw_text is not None
    )

    if st.button("Generate Summary", disabled=not generate_enabled):
        with st.spinner("Generating summary..."):
            # Determine the prompt to use
            if prompt_selection == "Custom":
                if not custom_prompt_input or not custom_prompt_input.strip():
                    st.warning("Please enter a custom prompt.")
                    st.stop()
                selected_prompt = custom_prompt_input
            else:
                selected_prompt = prompt_options[prompt_selection]

            # Use the stored raw_text for the query/context
            summary = get_ollama_response(
                st.session_state.vector_store,
                st.session_state.raw_text,  # Pass the full text
                selected_prompt,
            )
            if summary:
                st.subheader("Generated Summary")
                st.write(summary)
            else:
                # Error is already shown in get_ollama_response
                st.error("Failed to generate summary.")

    # Display info messages based on state
    if st.session_state.vector_store is None:
        st.info("Upload and process files to enable summary generation.")
    elif st.session_state.raw_text is None:
        # This covers cases where index loaded but text not processed,
        # or processing failed.
        st.info("Process uploaded files to enable summary generation.")


# Call main function directly
if __name__ == "__main__":
    main()
