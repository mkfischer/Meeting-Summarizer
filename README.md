# Meeting-Summarizer

This is a Streamlit application that summarizes meeting transcripts uploaded in PDF, TXT, or VTT format. It uses either the Ollama or OpenRouter language models to generate summaries in a structured format. A custom prompt option is also available for tailoring the output.

**Features**

*   Upload meeting transcripts in PDF, TXT, or VTT format.
*   Summarize transcripts using Ollama or OpenRouter language models.
*   Customize the summary format using a custom prompt.
*   Easy-to-use Streamlit interface.

**Installation**

1.  Clone this repository.
2.  Create a virtual environment using Python version > 3.9.
3.  Configure the `.env` file:
    *   For Ollama, ensure the `OLLAMA_MODEL` environment variable is set (e.g., `qwq:32b-preview-q4_K_M`).
    *   For OpenRouter, set the `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` environment variables.
4.  Run the application: `streamlit run app.py`
