# 🌍 Geo-Data Analyst: Self-Correcting Multi-Agent Mapping System

Welcome to the **Geo-Data Analyst**, an interactive web application built with Streamlit and LangGraph. This tool allows users to upload raw geospatial datasets (CSV) and interact with an AI agent in natural language to dynamically generate, refine, and display interactive Folium maps. 

Built for research and advanced data analysis, the system features a self-correcting multi-agent workflow that autonomously writes Python code, tests it, and refines it based on execution errors before presenting the final interactive map to the user.

## ✨ Key Features

* **Natural Language to Map:** Ask for complex spatial visualizations (e.g., "color the 20 southernmost points red") and get an interactive HTML map in seconds.
* **Self-Correcting Execution Loop:** Uses `langchain-experimental` PythonREPL to securely execute generated code. If the code fails, a Refiner agent automatically catches the traceback and rewrites the code until it succeeds.
* **RAG-Powered Code Retrieval:** Utilizes FAISS and HuggingFace embeddings to retrieve proven Folium map code snippets from a local vector store, anchoring the LLM's logic and syntax.
* **Conversational Interface:** Full chat history management with the ability to rename, delete, and switch between multiple active mapping sessions.
* **Automated Data Cleaning:** The agent is instructed to dynamically clean and filter only the active columns requested in the query to prevent global data loss.
* **Research Logging:** Automatically logs user prompts, execution times, error rates, generated code, and model responses to a backend database (e.g., Supabase) for analysis.

## 🧠 System Architecture

The core of the application is a StateGraph built with **LangGraph**. The workflow consists of:

1. **Retriever Node:** Queries a local FAISS database for contextual code examples based on the user's prompt.
2. **Agent Node:** Uses a large language model (e.g., TableGPT2-7B) to draft Pandas data manipulation and Folium mapping code. Includes explicit chain-of-thought planning.
3. **Execution Node:** Validates the code structure and runs the Python script. If successful, it captures the generated `map.html`.
4. **Refiner Node:** If the code fails execution, the error traceback is routed to a Refiner agent which patches the logic/syntax and sends it back for execution.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/geo-data-analyst.git
   cd geo-data-analyst
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   Ensure you have a `requirements.txt` file containing `streamlit`, `pandas`, `langgraph`, `langchain-openai`, `langchain-experimental`, `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `folium`, `h3`, and `supabase`.
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

The application relies on Streamlit Secrets for environment variables. Create a `.streamlit/secrets.toml` file in the root of your project:

```toml
# .streamlit/secrets.toml
CLUSTER_URL = "your_llm_cluster_endpoint_here"
MODEL_NAME = "your_model_name_here"

# If using Supabase for research logging:
SUPABASE_URL = "your_supabase_project_url"
SUPABASE_KEY = "your_supabase_anon_key"
```

*Note: The app currently initializes `ChatOpenAI` pointing to a custom cluster URL (useful for local or hosted open-source models).*

## 🚀 Usage

Start the Streamlit server:

```bash
streamlit run app_2.py
```

1. **Upload Data:** Use the sidebar to upload a CSV file containing geospatial data (latitude, longitude, and attributes).
2. **Chat:** Type a mapping request in the chat input.
3. **Interact:** The agent will draft the code, refine it if necessary, and render the interactive Folium map directly in the chat interface!

## 🗄️ Database & Vector Store Setup
* **Vector Store:** The app expects a compiled FAISS index populated from `vector_store/map_code_repository.json`. Ensure this exists so the Retriever node functions correctly.
* **Logging:** Set up a `session_logs` table in Supabase with a `JSONB` column named `log_data` to automatically capture research metrics.
