# streamlit_multiagent_displayer.py
import streamlit as st
import pandas as pd
import os
import re
import time
import json
import uuid
import operator
import traceback
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
from supabase import create_client



# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CLUSTER_URL    = st.secrets.get("CLUSTER_URL")
MODEL_NAME     = st.secrets.get("MODEL_NAME")
#DATA_FILE_PATH = "current_data.csv"
LOG_FILE       = "session_log.json"

st.set_page_config(page_title="Geo-Data Analyst", page_icon="🌍", layout="wide")
st.title("🌍 Geospatial Analyst (Self-Correcting)")


# ---------------------------------------------------------------------------
# LOGGING HELPER SUPABASE
# ---------------------------------------------------------------------------

def append_log(log_entry: dict):
    # log_entry is already the flat dictionary you showed me!
    # Its keys perfectly match the SQL columns we just created.
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        supabase = create_client(url, key)
        
        # Insert the dictionary directly; Supabase will map the keys to the columns
        supabase.table("session_logs").insert(log_entry).execute()
    except Exception as e:
        st.error(f"Supabase Error: {e}")


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
    if "user_session_id" not in st.session_state:
        st.session_state["user_session_id"] = str(uuid.uuid4())[:8]
    if "participant_id" not in st.session_state:
        st.session_state["participant_id"] = st.query_params.get("pid", "")
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
        _new_conversation()
    if "active_conv_id" not in st.session_state:
        convs = st.session_state["conversations"]
        if convs:
            st.session_state["active_conv_id"] = list(convs.keys())[-1]
        else:
            _new_conversation()
    if "last_uploaded_file" not in st.session_state:
        st.session_state["last_uploaded_file"] = None

_ensure_state()

USER_SESSION_ID = st.session_state["user_session_id"]
DATA_FILE_PATH  = f"data_{USER_SESSION_ID}.csv"
MAP_FILE_PATH   = f"map_{USER_SESSION_ID}.html"


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
    st.text_input(
        "Participant ID",
        key="participant_id",
        placeholder="e.g., P01 (optional)",
    )

    # ── Data Centre ─────────────────────────────────────────────────────────
    st.header("📂 Data Centre")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    data_summary  = "No data loaded."

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

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

            # --- User-Facing Welcome Message (runs once per new file) ---
            if st.session_state.get("last_uploaded_file") != uploaded_file.name:
                st.session_state["last_uploaded_file"] = uploaded_file.name

                md_table = "| Column Name | Data Type | Example Values |\n|---|---|---|\n"
                for col in df.columns:
                    readable_type = (
                        "Numeric"
                        if pd.api.types.is_numeric_dtype(df[col])
                        else "Text/Categorical"
                    )
                    samples = df[col].dropna().astype(str).unique()[:3]
                    sample_str = (
                        ", ".join(samples).replace("|", "\\|").replace("\n", " ")
                        if len(samples) > 0
                        else "N/A"
                    )
                    md_table += f"| **{col}** | {readable_type} | {sample_str} |\n"

                welcome_msg = (
                    f"👋 **Hello! Thanks for using the Geospatial Analyst.**\n\n"
                    f"I've successfully loaded your dataset (`{uploaded_file.name}`) containing **{len(df)} rows**. "
                    f"To help me build the most accurate map, please base your requests on the columns available below:\n\n"
                    f"{md_table}\n\n"
                    f"**What kind of map would you like to build?**"
                )

                active_id = st.session_state["active_conv_id"]
                st.session_state["conversations"][active_id]["history"].append(
                    {"role": "assistant", "content": welcome_msg}
                )
                st.rerun()

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


def retriever_node(state: AgentState):
    if vector_store is None:
        return {"code_context": "No context available."}

    user_query = state["messages"][-1].content
    
    # Retrieve top 1 clean match to avoid context clutter
    matched_docs = vector_store.similarity_search(user_query, k=1)
    
    code_context = matched_docs[0].metadata.get("code", "") if matched_docs else ""
    return {"code_context": code_context}


def agent_node(state):
    messages = state["messages"]

    system_prompt = f"""
    You are a strictly specialised Geospatial Python Analyst.

    DATA CONTEXT:
    {data_summary}

    RETRIEVED CODE EXAMPLES (Use ONLY for syntax logic, DO NOT copy file names or save calls):
    {state.get("code_context", "None available.")}

    TASK:
    Create a geographic map in Python based on the user request. If the user asks for a non-map visualization (e.g., bar chart, pie chart), politely refuse and do not generate code.
    
    VAGUE REQUEST & PROXY MAPPING RULES:
    1. PROXY INFERENCE: Users may ask for concepts not explicitly named in the columns (e.g., asking for "burned land area", "fire severity", or "most affected cities" when only point focus records exist). You MUST infer that frequency/count of points per category (e.g., city, region) serves as the proxy for these metrics.
    2. AGGREGATION TO MAPS: When asked for "top N cities" or rankings, calculate the top N using Pandas (`value_counts()` or `groupby()`), filter the original DataFrame to keep ONLY those points/cities, and plot those points or their centroids on the Folium map.

    FOLIUM MAP CATEGORY RULES:
    1. Points/Markers (DEFAULT): ALWAYS use `folium.Marker` with a `popup` label, EXCEPT when plotting a continuous color gradient (see Strict Rules 4).
    2. Clusters/Hotspots: Use `folium.plugins.MarkerCluster` ONLY if the user asks for "clusters" or "hotspots".
    3. Heatmaps: Use `folium.plugins.HeatMap` ONLY if explicitly requested.
    4. Spatial Grids: Use `h3` library for hexagons.

    TARGETED (QUERY-SCOPED) DATA CLEANING RULES:
    1. IDENTIFY ACTIVE COLUMNS: Determine ONLY the specific columns needed.
    2. SANITIZE SPATIAL DATA: Convert ONLY active lat/lon using `pd.to_numeric(df[col], errors='coerce')` and drop NaNs/zeros.
    3. SANITIZE ACTIVE ATTRIBUTE COLUMNS ONLY: Filter missing values for active columns only.
    4. NO GLOBAL DROPPING: NEVER use `df.dropna()` across the whole DataFrame.

    STRICT RULES:
    1. SCOPE: ONLY generate geographic maps.
    2. DATA: Always load data from '{DATA_FILE_PATH}'.
    3. CONDITIONAL STYLING: If styling depends on data conditions (e.g., "southernmost 20", "highest values"), you must first create a new categorical column or boolean mask in Pandas to isolate those specific rows BEFORE building the Folium map.
    4. COLOR GRADIENTS: ONLY when the user explicitly asks for a gradient/scale of colors, use `import branca.colormap as cm` with `cm.LinearColormap` and `folium.CircleMarker` (passing `color=cor, fill=True, fill_color=cor` AND `popup=...`). Convert datetime columns to numeric values (such as elapsed days) before passing them to the colormap, and attach the legend with `m.add_child(colormap)`.
    5. DISCRETE COLORS: If the user asks for different discrete colors (e.g., different colors for each biome, class, etc.), DO NOT use `branca.colormap`. Use `folium.Marker` and map categories to standard Folium color names ("red", "darkred", "lightred", "orange", "beige", "green", "darkgreen", "lightgreen", "blue", "darkblue", "lightblue", "cadetblue", "purple", "darkpurple", "pink", "white", "lightgray", "gray", "black") via `icon=folium.Icon(color=color_name)` with `popup=...`.
    6. POPUPS/CAPTIONS (MANDATORY): Every `folium.Marker` and `folium.CircleMarker` MUST include a `popup=` argument displaying the relevant queried attribute values (e.g., `popup=f"Risk: {{row['fire_risk']}}"`).
    7. BASE TILES: ALWAYS use the default OpenStreetMap tiles (e.g., use `folium.Map(...)` without the `tiles` argument). NEVER use 'CartoDB positron' or any other tile sets.
    8. OUTPUT FILE: ALWAYS save the map EXACTLY as '{MAP_FILE_PATH}'.
    9. FORMAT: Provide only a short explanation and the ```python ... ``` code block.

    PLANNING REQUIREMENT:
    Before writing Python code, write a 2-sentence plan inside `#` comments explaining your data filtering, proxy mapping, and color/layer assignment strategy.

    EXAMPLES:

    # Example 1: Proxy & Ranking
    # Plan: User asked for top 10 burned areas, but only point occurrences exist.
    # Count occurrences per city, isolate top 10 cities, filter points, and plot them.
    top_cities = df['city_column'].value_counts().head(10).index
    df_filtered = df[df['city_column'].isin(top_cities)].copy()
    # Build Folium map with df_filtered points...

    # Example 2: Conditional Styling
    # Plan: Isolate the 20 southernmost points by sorting latitude.
    # Color them red and all others blue using a new DataFrame column.
    bottom_20 = df.nsmallest(20, 'latitude')
    df['marker_color'] = 'blue'
    df.loc[bottom_20.index, 'marker_color'] = 'red'

    # Example 3: Continuous Datetime Color Gradient
    # Plan: Parse the date column and convert it to numeric elapsed days from the oldest record.
    # Apply a yellow-to-red LinearColormap to CircleMarkers and attach the legend.
    import branca.colormap as cm
    df['dt'] = pd.to_datetime(df['date_column'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'dt']).copy()
    df['days'] = (df['dt'] - df['dt'].min()).dt.total_seconds() / 86400.0
    colormap = cm.LinearColormap(colors=['yellow', 'orange', 'red'], vmin=df['days'].min(), vmax=df['days'].max())
    colormap.caption = 'Days Since Oldest Record'
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)
    for _, row in df.iterrows():
        cor = colormap(row['days'])
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            popup=f"Date: {{row['date_column']}}",
            radius=6,
            color=cor,
            fill=True,
            fill_color=cor,
            fill_opacity=0.7
        ).add_to(m)
    m.add_child(colormap)
    m.save('{MAP_FILE_PATH}')
    """

    # Pass all messages exactly as they are. The HumanMessage containing the error 
    # will act as the most recent prompt instructing the LLM to fix the code.
    msgs_for_llm = [SystemMessage(content=system_prompt)] + messages

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
    DATA SUMMARY:
    {data_summary}

    CRITICAL CHECKLIST:
    1. TARGETED CLEANING: Did it clean/filter ONLY the columns involved in the current request (e.g., active coordinates and active attribute columns)?
    2. NO GLOBAL LOSS: Did it avoid global `df.dropna()` or dropping rows based on unused columns?
    3. STYLING, MARKERS & POPUPS: Use `folium.Marker` by default; use `folium.CircleMarker` with `branca.colormap.LinearColormap` ONLY when a continuous color gradient is explicitly requested (ensuring `c = colormap(numeric_val)` is passed to `color=c, fill=True, fill_color=c` and `m.add_child(colormap)` is attached). Ensure EVERY `Marker` or `CircleMarker` includes an informative `popup=` argument.
    4. OUTPUT FILE: Ensure the map is saved EXACTLY as '{MAP_FILE_PATH}' (replace plt.show(), .show(), or any other output filenames).
    5. DATA INTEGRITY: Ensure it loads '{DATA_FILE_PATH}' and uses the exact column names from the summary.
    6. QUERY & SYNTAX: Ensure filters match the user query and fix all syntax or execution errors.
    
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

    if os.path.exists(MAP_FILE_PATH):
        os.remove(MAP_FILE_PATH)

    try:
        exec(code_to_run, {})

        if os.path.exists(MAP_FILE_PATH):
            timestamp = int(time.time())
            unique_html = f"map_{timestamp}.html"
            os.rename(MAP_FILE_PATH, unique_html)
            return {
                "messages": [HumanMessage(content=f"[HTML_READY:{unique_html}]", name="tool_success")],
                "retry_count": 0
            }
        else:
            error_details = (
                "Your previous code ran without errors, but did not create '{MAP_FILE_PATH}'. "
                "Ensure the map is saved using m.save('{MAP_FILE_PATH}')."
            )
            return {
                "messages": [HumanMessage(content=error_details, name="error")],
                "retry_count": current_retries + 1
            }
    except Exception:
        tb = traceback.format_exc()
        error_details = (
            f"Your previous code failed to generate the map.\n"
            f"Python Traceback:\n{tb}\n"
            f"Please fix the error and provide the corrected python code."
        )
        return {
            "messages": [HumanMessage(content=error_details, name="error")],
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
    follow_up_prefixes = (
        "same", "previous", "last", "chang", "instead", "also", "keep",
        "modif", "updat", "replac", "add", "remov", "without", "those", "these",
        "mesm", "anterior", "mud", "alter", "tro", "também", "mantenh", "ess"
    )
    is_follow_up = any(re.search(rf"\b{kw}", prompt.lower()) for kw in follow_up_prefixes)

    llm_messages: List[BaseMessage] = []
    uploaded_name = uploaded_file.name if uploaded_file is not None else None

    if is_follow_up and len(chat_history) >= 3:
        recent_turns = chat_history[-3:]  # Previous user + previous AI code + current user
    else:
        recent_turns = [chat_history[-1]]  # Standalone request: clean slate

    for turn in recent_turns:
        if turn["role"] == "user":
            llm_messages.append(HumanMessage(content=turn["content"]))
        else:
            clean = re.sub(r"\[HTML_READY:.*?\]", "", turn["content"])
            if uploaded_name:
                clean = clean.replace(uploaded_name, DATA_FILE_PATH)
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
        "user_session_id":        USER_SESSION_ID,
        "participant_id":         st.session_state.get("participant_id") or "anonymous",
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