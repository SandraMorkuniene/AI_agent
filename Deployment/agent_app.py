import streamlit as st
import os
import pandas as pd
import numpy as np
import ast
from typing import Dict, Any, List, Optional, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# --- TOOL FUNCTIONS ---
def fill_nulls_with_median(df): return df.fillna(df.median(numeric_only=True))
def normalize_missing_values(df): return df.replace(["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""], np.nan)
def drop_nulls(df): return df.dropna()
def standardize_column_names(df): df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_'); return df
def remove_duplicates(df): return df.drop_duplicates()
def convert_dtypes(df): return df.convert_dtypes()
def standardize_booleans(df):
    bool_map = {"yes": True, "no": False, "1": True, "0": False}
    for col in df.columns:
        if df[col].astype(str).str.lower().isin(bool_map.keys()).any():
            df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(df[col])
    return df

TOOLS = {
    "drop_nulls": drop_nulls,
    "fill_nulls_with_median": fill_nulls_with_median,
    "normalize_missing_values": normalize_missing_values,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "standardize_booleans": standardize_booleans
}

# --- LLM Setup ---
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- Generate summary of columns ---
def generate_column_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for col in df.columns:
        data = df[col]
        summary.append({
            "Column": col,
            "Type": str(data.dtype),
            "Missing Values": data.isnull().sum(),
            "Missing (%)": round(data.isnull().mean() * 100, 2),
            "Unique": data.nunique(),
            "Min": data.min() if pd.api.types.is_numeric_dtype(data) else "",
            "Max": data.max() if pd.api.types.is_numeric_dtype(data) else "",
            "Sample Values": ', '.join(map(str, data.dropna().unique()[:5]))
        })
    return pd.DataFrame(summary)

# --- LLM Suggest Fixes ---
def suggest_fixes(df: pd.DataFrame) -> List[str]:
    instruction = f"""
You are a data cleaning assistant. Here's a preview of the dataset:
{df.head().to_string()}

Suggest a list of likely cleaning steps from this set: {list(TOOLS.keys())}.
Respond with a Python list of tool names only.
"""
    response = llm.invoke(instruction).content.strip()
    try:
        return ast.literal_eval(response)
    except:
        return []

# --- Agent State ---
class AgentState(TypedDict):
    df: pd.DataFrame
    log: List[str]
    actions_taken: List[str]
    next_action: Optional[str]
    step_count: int

def run_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    df = state["df"]
    tool_func = TOOLS.get(tool_name)

    if not tool_func:
        state["log"].append(f"❌ Tool '{tool_name}' not found.")
        return state

    try:
        new_df = tool_func(df)
        state["df"] = new_df
        state["actions_taken"].append(tool_name)
        state["log"].append(f"✅ Ran tool: {tool_name}")
    except Exception as e:
        state["log"].append(f"❌ Failed to run tool '{tool_name}': {e}")
    return state

def build_graph(decider_func):
    builder = StateGraph(AgentState)
    builder.add_node("LLMDecision", decider_func)
    for tool_name in TOOLS:
        builder.add_node(tool_name, lambda state, t=tool_name: run_tool(state, t))
    builder.add_conditional_edges("LLMDecision", lambda x: x["next_action"], {
        **{name: name for name in TOOLS}, "end": END
    })
    for tool_name in TOOLS:
        builder.add_edge(tool_name, "LLMDecision")
    builder.set_entry_point("LLMDecision")
    return builder.compile()

def run_agent_pipeline(df: pd.DataFrame, allowed_tools: List[str], extra_instructions: str = ""):
    def decider(state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        done = state["actions_taken"]
        step_count = state["step_count"]

        if step_count > 10:
            state["log"].append("⚠️ Max cleaning steps reached. Ending.")
            state["next_action"] = "end"
            return state

        instruction = f"""
You are a data cleaning agent. The dataset preview is:
{df.head().to_string()}

Already applied: {done}
Allowed tools: {allowed_tools}
Extra user instructions: {extra_instructions}

Pick the next best tool (not already used), or return 'end'.
Respond ONLY with a tool name or 'end'.
"""
        tool = llm.invoke(instruction).content.strip().strip("'\"").lower()

        if tool in done or tool not in allowed_tools:
            if tool != "end":
                state["log"].append(f"⚠️ Ignored repeated/invalid tool: {tool}")
            state["next_action"] = "end"
        else:
            state["next_action"] = tool

        state["step_count"] += 1
        return state

    initial_state = {
        "df": df,
        "log": [],
        "actions_taken": [],
        "next_action": None,
        "step_count": 0
    }
    graph = build_graph(decider)
    final = graph.invoke(initial_state, config=RunnableConfig())
    return final["df"], final["log"]

# --- Streamlit UI ---
st.set_page_config(page_title="Interactive Data Cleaner", layout="wide")
st.title("🧠 Interactive Data Cleaner Agent")

if "df" not in st.session_state:
    st.session_state.df = None
if "suggested_tools" not in st.session_state:
    st.session_state.suggested_tools = []
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if st.button("🧹 Reset"):
    for key in ["df", "suggested_tools", "cleaned_df"]:
        st.session_state[key] = None
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
            st.subheader("📄 Original Data")
            st.dataframe(df.head(100), use_container_width=True)

            st.markdown("### 📊 Column Summary for Evaluation")
            summary_df = generate_column_summary_table(df)
            st.dataframe(summary_df, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

# --- Suggest Cleaning Steps ---
if st.session_state.df is not None:
    if st.button("🧠 Analyze & Suggest Cleaning Steps"):
        st.session_state.suggested_tools = suggest_fixes(st.session_state.df)

if st.session_state.suggested_tools:
    st.subheader("🔧 Suggested Cleaning Steps")
    st.code(st.session_state.suggested_tools)
    selected_tools = st.multiselect("Choose tools to apply", options=st.session_state.suggested_tools, default=st.session_state.suggested_tools)
    extra_input = st.text_input("Extra instructions for the agent (optional)")
    if st.button("🚀 Run Cleaner"):
        with st.spinner("Agent cleaning in progress..."):
            cleaned, log = run_agent_pipeline(st.session_state.df, selected_tools, extra_input)
            st.session_state.cleaned_df = cleaned
            st.session_state.log = log
        st.success("✅ Cleaning complete.")

# --- Show Cleaned Data ---
if st.session_state.cleaned_df is not None:
    st.subheader("📦 Final Cleaned Data")
    st.dataframe(st.session_state.cleaned_df.head(100), use_container_width=True)

    st.markdown("### 📊 Updated Column Summary")
    summary_df = generate_column_summary_table(st.session_state.cleaned_df)
    st.dataframe(summary_df, use_container_width=True)

    st.download_button("⬇ Download Cleaned CSV", st.session_state.cleaned_df.to_csv(index=False), "cleaned.csv", "text/csv")

    st.markdown("### 🗣️ Provide Feedback to Improve Cleaning")
    feedback = st.text_area("Still see issues? Describe them:")
    if st.button("🔁 Re-run With Feedback"):
        with st.spinner("Agent re-cleaning in progress..."):
            re_cleaned, re_log = run_agent_pipeline(st.session_state.cleaned_df, list(TOOLS.keys()), feedback)
            st.session_state.cleaned_df = re_cleaned
            st.session_state.log += re_log
        st.success("✅ Agent re-cleaning complete.")
        st.rerun()
