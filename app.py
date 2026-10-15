# streamlit_multiagent_displayer.py
import streamlit as st
import pandas as pd
import os
import re
import time
import json
import uuid
import operator
from datetime import datetime
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CLUSTER_URL    = st.secrets.get("CLUSTER_URL")
MODEL_NAME     = st.secrets.get("MODEL_NAME")
DATA_FILE_PATH = "current_data.csv"
LOG_FILE       = "session_log.json"

st.set_page_config(page_title="Geo-Data Analyst", page_icon="🌍", layout="wide")
st.title("🌍 Geospatial Analyst (Self-Correcting)")


# ---------------------------------------------------------------------------
# LOGGING HELPER
# ---------------------------------------------------------------------------
def append_log(entry: dict):
    """Append a structured log entry to session_log.json."""
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CONVERSATION STATE HELPERS
# ---------------------------------------------------------------------------
def _make_conv_id() -> str:
    return str(uuid.uuid4())[:8]


def _new_conversation():
    """Create a blank conversation and set it as active."""
    conv_id = _make_conv_id()
    st.session_state["conversations"][conv_id] = {
        "name": "New Chat",
        "history": [],
    }
    st.session_state["active_conv_id"] = conv_id


def _ensure_state():
    """Initialise session-state keys on the very first run."""
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
        _new_conversation()
    if "active_conv_id" not in st.session_state:
        convs = st.session_state["conversations"]
        if convs:
            st.session_state["active_conv_id"] = list(convs.keys())[-1]
        else:
            _new_conversation()


_ensure_state()


def _init_rename_state():
    if "rename_conv_id" not in st.session_state:
        st.session_state["rename_conv_id"] = None


def _start_rename(conv_id: str, current_name: str):
    st.session_state["rename_conv_id"] = conv_id
    st.session_state[f"rename_input_{conv_id}"] = current_name


def _save_rename(conv_id: str):
    key = f"rename_input_{conv_id}"
    new_name = st.session_state.get(key, "").strip()
    if new_name:
        st.session_state["conversations"][conv_id]["name"] = new_name[:80]
    st.session_state["rename_conv_id"] = None


def _cancel_rename():
    st.session_state["rename_conv_id"] = None


_init_rename_state()

# ---------------------------------------------------------------------------
# SIDEBAR — Data Centre + Conversation Manager
# ---------------------------------------------------------------------------
with st.sidebar:

    # ── Data Centre ─────────────────────────────────────────────────────────
    st.header("📂 Data Centre")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    data_summary  = "No data loaded."

    if uploaded_file is not None:
        try:
            # OLD CODE (Fails in Pandas 3.0+)
            # df = pd.read_csv(uploaded_file)
            # object_cols = df.select_dtypes(include=["object"]).columns
            # df[object_cols] = df[object_cols].astype(str)
            # df.fillna(0, inplace=True)

            # NEW CODE (Compatible with Pandas 3.0+)
            df = pd.read_csv(uploaded_file)

            # # Fill numeric NaNs with 0
            # num_cols = df.select_dtypes(include=["number"]).columns
            # df[num_cols] = df[num_cols].fillna(0)

            # # Fill text/object NaNs with empty string and cast to str
            # non_num_cols = df.select_dtypes(exclude=["number"]).columns
            # df[non_num_cols] = df[non_num_cols].fillna("").astype(str)
            df.to_csv(DATA_FILE_PATH, index=False)
            st.success(f"Loaded {len(df)} rows")

            buffer = [
                f"Filename: '{DATA_FILE_PATH}'",
                f"Total Rows: {len(df)}",
                "--- COLUMN ANALYSIS ---",
            ]
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    buffer.append(
                        f"- {col} (Numeric): [{df[col].min()} to {df[col].max()}]"
                    )
                else:
                    unique_vals = df[col].unique()
                    if len(unique_vals) < 15:
                        buffer.append(
                            f"- {col} (Categorical): [{', '.join(map(str, unique_vals))}]"
                        )
                    else:
                        buffer.append(
                            f"- {col} (Text): Example [{', '.join(map(str, unique_vals[:3]))}, ...]"
                        )
            data_summary = "\n".join(buffer)
            with st.expander("LLM View"):
                st.text(data_summary)
        except Exception as e:
            st.error(f"Error: {e}")

        # ── Conversation Manager ─────────────────────────────────────────────────
    st.divider()
    st.header("💬 Conversations")

    if st.button("➕ New Chat", use_container_width=True):
        _new_conversation()
        st.rerun()

    st.markdown("---")

    convs = st.session_state["conversations"]
    active_id = st.session_state["active_conv_id"]

    for conv_id, conv_data in list(convs.items()):
        is_active = (conv_id == active_id)
        is_renaming = (st.session_state.get("rename_conv_id") == conv_id)
        label = f"{'▶ ' if is_active else ''}{conv_data['name']}"

        col_name, col_ren, col_del = st.columns([4, 1, 1])

        with col_name:
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                label,
                key=f"sel_{conv_id}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state["active_conv_id"] = conv_id
                st.rerun()

        with col_ren:
            if st.button("✏️", key=f"ren_{conv_id}", help="Rename this conversation"):
                _start_rename(conv_id, conv_data["name"])
                st.rerun()

        with col_del:
            if st.button("🗑", key=f"del_{conv_id}", help="Delete this conversation"):
                del st.session_state["conversations"][conv_id]

                if st.session_state.get("rename_conv_id") == conv_id:
                    _cancel_rename()

                if active_id == conv_id:
                    remaining = list(st.session_state["conversations"].keys())
                    if remaining:
                        st.session_state["active_conv_id"] = remaining[-1]
                    else:
                        _new_conversation()

                st.rerun()

        if is_renaming:
            rename_key = f"rename_input_{conv_id}"
            if rename_key not in st.session_state:
                st.session_state[rename_key] = conv_data["name"]

            st.text_input(
                "Conversation name",
                key=rename_key,
                label_visibility="collapsed",
            )

            c_save, c_cancel = st.columns(2)

            with c_save:
                if st.button("Save", key=f"save_{conv_id}", use_container_width=True):
                    _save_rename(conv_id)
                    st.rerun()

            with c_cancel:
                if st.button("Cancel", key=f"cancel_{conv_id}", use_container_width=True):
                    _cancel_rename()
                    st.rerun()


# ---------------------------------------------------------------------------
# RESOLVE ACTIVE CONVERSATION
# ---------------------------------------------------------------------------
active_conv_id = st.session_state["active_conv_id"]
# chat_history is a reference into session_state — mutations persist automatically
chat_history   = st.session_state["conversations"][active_conv_id]["history"]


# ---------------------------------------------------------------------------
# LLM + REPL SETUP  (runs every Streamlit rerun, costs nothing)
# ---------------------------------------------------------------------------
llm  = ChatOpenAI(
    base_url=CLUSTER_URL,
    api_key="EMPTY",
    model=MODEL_NAME,
    temperature=0.1,
)
repl = PythonREPL()


# ---------------------------------------------------------------------------
# LANGGRAPH STATE
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    subtasks: List[str]        
    code_context: str
    retry_count: int  # Track failed attempts


# ---------------------------------------------------------------------------
# VECTOR DATABASE SETUP
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_store():
    # Load a lightweight, fast, local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"})
    
    # Load your JSON database
    try:
        with open("vector_store/map_code_repository.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        st.error("RAG Database not found. Please ensure 'map_code_repository.json' is in the directory.")
        return None

    # Convert to LangChain Documents
    documents = []
    for chunk in chunks:
        # We combine description and libraries for a rich semantic search string
        page_content = f"{chunk['description']} (Libraries: {', '.join(chunk['libraries'])})"
        # Store the actual code in metadata so we can retrieve it
        metadata = {"code": chunk["code"], "id": chunk["id"]}
        documents.append(Document(page_content=page_content, metadata=metadata))

    # Initialize the FAISS vector store
    return FAISS.from_documents(documents, embeddings)

vector_store = load_vector_store()



# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------

def decomposer_node(state: AgentState):
    # Grab the most recent user request
    user_query = state["messages"][-1].content
    
    prompt = f"""You are a Geospatial Architect. Break down the user's request into explicit, programmatic subtasks.
    User Request: {user_query}
    
    Respond with a numbered list of steps required to achieve this in Python. Do not write code."""
    
    # Ask the LLM to think through the problem
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Parse the text into a clean list of steps
    subtasks = [line for line in response.content.split('\n') if line.strip()]
    
    return {"subtasks": subtasks}


# def retriever_node(state: AgentState):
#     if vector_store is None:
#         return {"code_context": "No retrieved context available."}

#     user_query = state["messages"][-1].content
#     subtasks_str = " ".join(state.get("subtasks", []))
    
#     # Combine query and subtasks to form a highly specific search string
#     search_query = f"{user_query} {subtasks_str}"
    
#     # Retrieve the top 2 closest matches
#     matched_docs = vector_store.similarity_search(search_query, k=2)
    
#     # Format the retrieved code into a single context string to feed the Agent
#     context_blocks = []
#     for i, doc in enumerate(matched_docs):
#         context_blocks.append(f"--- Example {i+1} ---\n{doc.metadata['code']}")
        
#     code_context = "\n\n".join(context_blocks)
    
#     return {"code_context": code_context}

def retriever_node(state: AgentState):
    if vector_store is None:
        return {"code_context": "No context available."}

    user_query = state["messages"][-1].content
    
    # Retrieve top 1 clean match to avoid context clutter
    matched_docs = vector_store.similarity_search(user_query, k=1)
    
    code_context = matched_docs[0].metadata.get("code", "") if matched_docs else ""
    return {"code_context": code_context}

#PLAN:
#    {chr(10).join(state.get("subtasks", []))}

def agent_node(state):
    messages = state["messages"]

    error_context = ""
    last_msg = messages[-1]
    if hasattr(last_msg, "name") and last_msg.name in ["error", "refiner_error"]:
        error_context = (
            f"\n\nYOUR PREVIOUS CODE FAILED WITH:\n{last_msg.content}"
            "\nPlease fix the python code."
        )

    system_prompt = f"""
    You are a strictly specialised Geospatial Python Analyst.

    DATA CONTEXT:
    {data_summary}

    RETRIEVED CODE EXAMPLES (Use ONLY for syntax logic, DO NOT copy file names or save calls):
    {state.get("code_context", "None available.")}

    TASK:
    Create a geographic map in Python based on the user request. If the user asks for a non-map visualization (e.g., bar chart, pie chart), politely refuse and do not generate code.

    TARGETED (QUERY-SCOPED) DATA CLEANING RULES:
    1. IDENTIFY ACTIVE COLUMNS: Determine ONLY the specific columns needed for the current request (e.g., Latitude, Longitude, and the specific attribute column like 'period_of_day').
    2. SANITIZE SPATIAL DATA: Convert ONLY the active latitude/longitude columns using `pd.to_numeric(df[col], errors='coerce')` and filter out rows where coordinates are NaN or zero.
    3. SANITIZE ACTIVE ATTRIBUTE COLUMNS ONLY:
       - For active numeric/sorting columns: Convert using `pd.to_numeric(..., errors='coerce')` and filter out NaNs with 'df.dropna' for THAT column only.
       - For active categorical/color columns: Fill NaNs in THAT column with 'Unknown' or filter missing values for THAT column only.
    4. NO GLOBAL DROPPING: NEVER use `df.dropna()` across the whole DataFrame, and NEVER drop rows based on columns not requested by the user. Preserve as much data as possible.

    STRICT RULES:
    1. SCOPE: ONLY generate geographic maps (using libraries like folium, plotly, geopandas, cartopy, osmnx, basemap, keplergl or ipyleaflet).
    2. DATA: Always load data from '{DATA_FILE_PATH}'.
    3. OUTPUT FILE: ALWAYS save the map EXACTLY as 'map.html'.
    4. FORMAT: Provide only a short explanation and the ```python ... ``` code block.{error_context}
    """

    msgs_for_llm = [SystemMessage(content=system_prompt)]
    for m in messages:
        if m.type in ["human", "ai"] and (
            not hasattr(m, "name") or m.name not in ["error", "refiner_error"]
        ):
            msgs_for_llm.append(m)

    response = llm.invoke(msgs_for_llm)
    return {"messages": [response]}


def refiner_node(state):
    last_message = state["messages"][-1]

    code_match = re.search(r"```python(.*?)```", last_message.content, re.DOTALL)
    if not code_match:
        # No code block — polite refusal or other text; let it flow to END
        return {"messages": []}

    draft_code = code_match.group(1).strip()

    review_prompt = f"""
    You are a Senior Geospatial Code Reviewer. Check this code draft:
    ```python
    {draft_code}
    ```
    CRITICAL CHECKLIST:
    1. TARGETED CLEANING: Did it clean/filter ONLY the columns involved in the current request (e.g., active coordinates and active attribute columns)?
    2. NO GLOBAL LOSS: Did it avoid global `df.dropna()` or dropping rows based on unused columns?
    3. OUTPUT FILE: Ensure the map is saved EXACTLY as 'map.html' (replace plt.show(), .show(), or any other output filenames).
    4. DATA INTEGRITY: Ensure it loads '{DATA_FILE_PATH}' and uses the exact column names from the summary.
    5. QUERY & SYNTAX: Ensure filters match the user query and fix all syntax or execution errors.
    
    Return ONLY the improved code wrapped in ```python ... ```. Do not explain.
    """
    review_response = llm.invoke([HumanMessage(content=review_prompt)])

    refined_match = re.search(r"```python(.*?)```", review_response.content, re.DOTALL)
    if refined_match:
        refined_code = refined_match.group(1).strip()
        new_msg = AIMessage(
            content=f"Here is the code for your map:\n```python\n{refined_code}\n```"
        )
        return {"messages": [new_msg]}

    return {"messages": []}


def execute_node(state: AgentState):
    current_retries = state.get("retry_count", 0)
    
    code_to_run = None
    for msg in reversed(state["messages"]):
        if msg.type == "ai" and "```python" in msg.content:
            code_match = re.search(r"```python(.*?)```", msg.content, re.DOTALL)
            if code_match:
                code_to_run = code_match.group(1).strip()
                break

    if not code_to_run:
        return {
            "messages": [HumanMessage(content="Execution Error: No python code block found.", name="error")],
            "retry_count": current_retries + 1
        }

    if os.path.exists("map.html"):
        os.remove("map.html")

    try:
        # repl.run returns stdout/stderr/traceback as a string
        execution_output = repl.run(code_to_run)

        if os.path.exists("map.html"):
            timestamp = int(time.time())
            unique_html = f"map_{timestamp}.html"
            os.rename("map.html", unique_html)
            return {
                "messages": [HumanMessage(content=f"[HTML_READY:{unique_html}]", name="tool_success")],
                "retry_count": 0  # Reset on success
            }
        else:
            # Feed the actual Python error/traceback back to TableGPT2
            error_details = (
                f"Execution Error: 'map.html' was not generated.\n"
                f"Python REPL Output / Traceback:\n{execution_output}"
            )
            return {
                "messages": [HumanMessage(content=error_details, name="error")],
                "retry_count": current_retries + 1
            }
    except Exception as e:
        return {
            "messages": [HumanMessage(content=f"Execution Error: {e}", name="error")],
            "retry_count": current_retries + 1
        }


MAX_RETRIES = 2  # Allows initial run + 2 retry attempts

def router(state: AgentState):
    last_msg = state["messages"][-1]
    retries = state.get("retry_count", 0)

    if hasattr(last_msg, "name") and last_msg.name == "error":
        if retries <= MAX_RETRIES:
            return "agent"
        else:
            # Stop loop and return to user
            return END
    return END


# ---------------------------------------------------------------------------
# COMPILE GRAPH
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)
#workflow.add_node("decomposer", decomposer_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("agent", agent_node) # Existing[cite: 1]
workflow.add_node("refiner", refiner_node) # Existing[cite: 1]
workflow.add_node("execute_code", execute_node) # Existing[cite: 1]

#workflow.set_entry_point("decomposer")
workflow.set_entry_point("retriever")
#workflow.add_edge("decomposer", "retriever")
workflow.add_edge("retriever", "agent")
workflow.add_edge("agent", "refiner") # Existing[cite: 1]
workflow.add_edge("refiner", "execute_code") # Existing[cite: 1]
workflow.add_conditional_edges("execute_code", router) # Existing[cite: 1]
app_graph = workflow.compile()


# ---------------------------------------------------------------------------
# RENDER EXISTING HISTORY
# ---------------------------------------------------------------------------
def render_message(role: str, content: str):
    """Render a single message, expanding any embedded HTML map tag."""
    with st.chat_message(role):
        html_match = re.search(r"\[HTML_READY:(.*?)\]", content)
        if html_match:
            try:
                with open(html_match.group(1), "r") as f:
                    components.html(f.read(), height=500)
            except Exception:
                st.error("Map file unavailable.")
            content = content.replace(html_match.group(0), "")
        st.markdown(content.strip())


for msg in chat_history:
    render_message(msg["role"], msg["content"])


# ---------------------------------------------------------------------------
# HANDLE NEW USER INPUT
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask for a geographic map..."):

    # ── Name the conversation after its first user message ─────────────────
    if st.session_state["conversations"][active_conv_id]["name"] == "New Chat":
        short_name = prompt[:35] + ("…" if len(prompt) > 35 else "")
        st.session_state["conversations"][active_conv_id]["name"] = short_name

    # Persist and display user turn
    chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ── Build clean LLM message history (no [HTML_READY] tags) ─────────────
    llm_messages: List[BaseMessage] = []
    for turn in chat_history:
        if turn["role"] == "user":
            llm_messages.append(HumanMessage(content=turn["content"]))
        else:
            clean = re.sub(r"\[HTML_READY:.*?\]", "", turn["content"])
            llm_messages.append(AIMessage(content=clean.strip()))

    inputs = {"messages": llm_messages, "retry_count": 0}

    # ── Metrics to track for the log ───────────────────────────────────────
    metrics = {
        "refinement_activations": 0,
        "code_execution_errors":  0,
        "system_errors":          0,
        "generated_code":         "",
    }

    final_code_msg = ""
    html_tag       = ""
    # Track the last error string for UI rendering
    last_error_details = ""
    t_start        = time.time()

    
    with st.chat_message("assistant"):
        with st.status("Analysing and Generating Map…", expanded=True) as status:

            for event in app_graph.stream(inputs, config={"recursion_limit": 25}):
                for key, value in event.items():
                    if "messages" not in value or not value["messages"]:
                        continue
                    msg = value["messages"][-1]

                    if key == "agent":
                        st.write("🧠 Agent is drafting code…")

                    elif key == "refiner":
                        metrics["refinement_activations"] += 1
                        st.write("🔍 Refiner is checking code…")
                        if msg.type == "ai":
                            final_code_msg = msg.content
                            code_match = re.search(
                                r"```python(.*?)```", msg.content, re.DOTALL
                            )
                            if code_match:
                                metrics["generated_code"] = code_match.group(1).strip()

                    elif key == "execute_code":
                        if hasattr(msg, "name") and msg.name == "tool_success":
                            st.write("✅ Execution successful!")
                            html_tag = msg.content
                        elif hasattr(msg, "name") and msg.name == "error":
                            metrics["code_execution_errors"] += 1
                            last_error_details = msg.content  # Store actual error
                            st.error("⚠️ Execution failed. Agent is retrying…")

            status.update(label="Process Complete", state="complete")

    # Display detailed error if execution failed completely
    if not html_tag and metrics["code_execution_errors"] > 0:
        final_code_msg += (
            f"\n\n⚠️ **Execution Failed:** Could not generate map.\n"
            f"**Error details:**\n```\n{last_error_details}\n```"
        )

    # ── Measure total time ─────────────────────────────────────────────────
    response_time = round(time.time() - t_start, 2)

    # ── Build final response string ─────────────────────────────────────────
    final_response = (
        f"{final_code_msg}\n\n{html_tag}" if html_tag else final_code_msg
    )

    # ── Persist assistant turn ─────────────────────────────────────────────
    chat_history.append({"role": "assistant", "content": final_response})

    # ── Render the result ──────────────────────────────────────────────────
    render_content = final_response
    if html_tag:
        html_match = re.search(r"\[HTML_READY:(.*?)\]", render_content)
        if html_match:
            with st.chat_message("assistant"):
                try:
                    with open(html_match.group(1), "r") as f:
                        components.html(f.read(), height=500)
                except Exception:
                    st.error("Map file unavailable.")
            render_content = render_content.replace(html_match.group(0), "")
    if render_content.strip():
        with st.chat_message("assistant"):
            st.markdown(render_content.strip())

    # Show which execution error was trigged
    if not html_tag and metrics["code_execution_errors"] > 0:
        # If code execution failed completely after retries, print the error to the chat UI
        last_error = inputs["messages"][-1].content if inputs["messages"] else "Unknown error"
        final_response += f"\n\n⚠️ **Execution Failed:** The agent could not produce a working map. Error details:\n```\n{last_error}\n```"


    # ── Write system log entry ─────────────────────────────────────────────
    log_entry = {
        "timestamp":              datetime.now().isoformat(timespec="seconds"),
        "conversation_id":        active_conv_id,
        "conversation_name":      st.session_state["conversations"][active_conv_id]["name"],
        "prompt":                 prompt,
        "generated_code":         metrics["generated_code"],
        "refinement_activations": metrics["refinement_activations"],
        "code_execution_errors":  metrics["code_execution_errors"],
        "system_errors":          metrics["system_errors"],
        "response_time_seconds":  response_time,
    }
    append_log(log_entry)