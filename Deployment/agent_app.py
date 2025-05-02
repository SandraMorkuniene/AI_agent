import streamlit as st
import os
import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, TypedDict, List, Optional

### TOOLS ###

def fill_nulls_with_median(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""], np.nan)
    
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()
    
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.convert_dtypes()

def standardize_booleans(df: pd.DataFrame) -> pd.DataFrame:
    bool_map = {"yes": True, "no": False, "1": True, "0": False}
    for col in df.columns:
        if df[col].astype(str).str.lower().isin(bool_map.keys()).any():
            df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(df[col])
    return df

def generate_eda_summary(df: pd.DataFrame) -> str:
    buffer = []
    buffer.append("Shape:\n" + str(df.shape))
    buffer.append("\nMissing Values:\n" + str(df.isnull().sum()))
    buffer.append("\nSummary Stats:\n" + df.describe().to_string())
    return "\n\n".join(buffer)

TOOLS = {
    "drop_nulls": drop_nulls,
    "fill_nulls_with_median": fill_nulls_with_median,
    "normalize_missing_values": normalize_missing_values,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "standardize_booleans": standardize_booleans,
    "generate_eda_summary": generate_eda_summary
}

# Add OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

llm = ChatOpenAI(model="gpt-4", temperature=0)

### LLM Decision Logic ###
def decide_next_action(state: Dict[str, Any]) -> str:
    df = state["df"]
    cleaned_steps = state["actions_taken"]
    feedback = state.get("user_feedback", "")

    instruction = f'''
    You are a data cleaning agent. You already applied: {cleaned_steps}.
    Dataset preview:
    {df.head().to_string()}

    User feedback: "{feedback}"

    Available tools: {list(TOOLS.keys())}

    Choose the next best tool to apply. 
    Only choose tools not yet applied unless the user explicitly requested it. 
    Respond ONLY with a tool name, or 'end' if cleaning is complete.
    '''

    response = llm.invoke(instruction).content.strip().lower().strip('"').strip("'")
    if response in cleaned_steps and feedback.strip() == "":
        return "end"
    return response

def run_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    df = state["df"]
    tool_func = TOOLS.get(tool_name)
    state["step_count"] += 1

    if tool_func is None:
        state["log"].append(f"❌ Tool '{tool_name}' not found in registry.")
        return state

    try:
        new_df = tool_func(df)
        state["df"] = new_df
        state["actions_taken"].append(tool_name)
        state["log"].append(f"✅ Ran tool: {tool_name}")
    except Exception as e:
        state["log"].append(f"❌ Failed to run tool '{tool_name}': {e}")

    return state

class AgentState(TypedDict):
    df: pd.DataFrame
    log: List[str]
    actions_taken: List[str]
    next_action: Optional[str]
    user_feedback: Optional[str]
    step_count: int

### Generate Column Summaries ###
def generate_column_summaries(df: pd.DataFrame) -> str:
    summary = []
    
    for column in df.columns:
        col_data = df[column]
        col_summary = f"**Column: {column}**\n"
        
        # Data type
        col_summary += f"  - Type: {col_data.dtype}\n"
        
        # Missing values
        missing_count = col_data.isnull().sum()
        col_summary += f"  - Missing values: {missing_count} ({(missing_count / len(col_data)) * 100:.2f}%)\n"
        
        # Unique values (only display up to 10 unique values)
        unique_vals = col_data.nunique()
        col_summary += f"  - Unique values: {unique_vals}\n"
        if unique_vals <= 10:
            col_summary += f"    - Values: {', '.join(col_data.dropna().unique().astype(str))}\n"
        
        # For numeric columns, add summary statistics
        if pd.api.types.is_numeric_dtype(col_data):
            stats = col_data.describe().to_dict()
            col_summary += f"  - Summary Stats: {stats}\n"
        
        summary.append(col_summary)
    
    return "\n".join(summary)

### Build LangGraph ###
def build_graph(llm_decision_func):
    builder = StateGraph(AgentState)

    builder.add_node("LLMDecision", llm_decision_func)

    # Tool runner node gets selected by LLM
    for tool_name in TOOLS.keys():
        builder.add_node(tool_name, lambda state, t=tool_name: run_tool(state, t))

    builder.add_conditional_edges("LLMDecision", lambda x: x["next_action"], {
        **{name: name for name in TOOLS.keys()},
        "end": END
    })

    # Loop back after each tool execution
    for tool_name in TOOLS.keys():
        builder.add_edge(tool_name, "LLMDecision")

    builder.set_entry_point("LLMDecision")
    return builder.compile()

### Run the Agent Pipeline ###
def run_agent_pipeline(df: pd.DataFrame, tools_list: List[str], feedback: str = ""):
    initial_state = {
        "df": df,
        "log": [],
        "actions_taken": [],
        "next_action": None,
        "user_feedback": feedback,
        "step_count": 0
    }

    def wrapper(state):
        if state["step_count"] > 15:
            state["log"].append("⚠️ Max cleaning steps reached. Ending.")
            state["next_action"] = "end"
            return state

        action = decide_next_action(state)
        state["next_action"] = action
        return state

    graph = build_graph(wrapper)
    final = graph.invoke(initial_state, config=RunnableConfig())
    return final["df"], final["log"]

### Streamlit Frontend ###

st.set_page_config(page_title="Smart Data Cleaning Agent", layout="wide")
st.title("🧠 Smart Data Cleaning Agent")

# --- Session Initialization ---
init_keys = {
    "df": None,
    "log": [],
    "cleaned_df": None,
    "step_selection": [],
    "clean_log": [],
}
for k, v in init_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Clear Session ---
if st.button("🧹 Clear Session"):
    for k in init_keys:
        st.session_state[k] = init_keys[k]
    st.rerun()

# --- Upload CSV ---
file = st.file_uploader("📂 Upload your CSV", type=["csv"])
if file:
    try:
        df = pd.read_csv(file)
        if df.empty:
            st.error("⚠️ Uploaded CSV is empty.")
        else:
            st.session_state.df = df
            st.write("📄 **Original Data**")
            st.dataframe(df, use_container_width=True)
            
            # Generate column summaries and display as context for user evaluation
            column_summary = generate_column_summaries(df)
            st.markdown("### Column Summary for Evaluation:")
            st.markdown(column_summary)

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

# --- Run Cleaning ---
if st.session_state.df is not None and st.button("🚀 Run Smart Cleaning Agent"):
    try:
        with st.spinner("Agent is working..."):
            cleaned_df, log = run_agent_pipeline(st.session_state.df, list(TOOLS.keys()))

        st.session_state.cleaned_df = cleaned_df
        st.session_state.log = log
        st.session_state.step_selection = [l.startswith("✅ Ran tool: ") for l in log]

        st.success("✅ Agent cleaning complete. See steps below.")
    except Exception as e:
        st.error(f"❌ Error during cleaning process: {e}")

# --- Show Step-by-Step Results ---
if st.session_state.log:
    st.subheader("🔍 Step-by-Step Preview of Cleaning")
    preview_df = st.session_state.df.copy()

    for i, step in enumerate(st.session_state.log):
        st.markdown(step)
        if step.startswith("✅ Ran tool: "):
            tool = step.replace("✅ Ran tool: ", "").strip()
            try:
                preview_df = TOOLS[tool](preview_df)
                st.dataframe(preview_df, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Failed to apply tool '{tool}': {e}")

# --- Final Output ---
if st.session_state.cleaned_df is not None:
    st.subheader("📦 Final Cleaned Data (Downloadable)")
    st.dataframe(st.session_state.cleaned_df, use_container_width=True)

    if st.session_state.cleaned_df.empty:
        st.warning("⚠️ Final result is empty. Try unchecking some cleaning steps.")
