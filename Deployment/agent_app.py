import streamlit as st
import os
import pandas as pd
import numpy as np
import ast
from typing import Dict, Any, List, Optional, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

### TOOL FUNCTIONS ###
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
def generate_eda_summary(df):
    buffer = ["Shape:\n" + str(df.shape),
              "\nMissing Values:\n" + str(df.isnull().sum()),
              "\nSummary Stats:\n" + df.describe().to_string()]
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

### LLM SETUP ###
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(model="gpt-4", temperature=0)

### LLM Suggestions ###
def suggest_fixes(df: pd.DataFrame) -> List[str]:
    instruction = f"""
You are a data cleaning assistant. Here is a preview of the dataset:
{df.head().to_string()}

From this, suggest a list of necessary cleaning steps from this set: {list(TOOLS.keys())}.
Respond with a Python list of tool names only.
"""
    response = llm.invoke(instruction).content.strip()
    try:
        return ast.literal_eval(response)
    except:
        return []

### AGENT EXECUTION ###
class AgentState(TypedDict):
    df: pd.DataFrame
    log: List[str]
    actions_taken: List[str]
    next_action: Optional[str]

def run_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    df = state["df"]
    tool_func = TOOLS.get(tool_name)
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
        instruction = f"""
You're a cleaning agent. Done steps: {done}.
Extra user instructions: {extra_instructions}
Pick next from: {allowed_tools}.
Dataset preview:
{df.head().to_string()}
Respond with one tool name or 'end'.
"""
        response = llm.invoke(instruction).content.strip().strip("'\"")
        return {**state, "next_action": response.lower()}

    initial_state = {"df": df, "log": [], "actions_taken": [], "next_action": None}
    graph = build_graph(decider)
    final = graph.invoke(initial_state, config=RunnableConfig())
    return final["df"], final["log"]

### STREAMLIT UI ###
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

file = st.file_uploader("📂 Upload your CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.session_state.df = df
    st.subheader("📄 Preview Uploaded Data")
    st.dataframe(df.head(100))

if st.session_state.df is not None:
    if st.button("🧠 Analyze & Suggest Cleaning Steps"):
        st.session_state.suggested_tools = suggest_fixes(st.session_state.df)

if st.session_state.suggested_tools:
    st.subheader("🔧 Suggested Cleaning Steps")
    selected_tools = st.multiselect("Choose tools to apply", options=st.session_state.suggested_tools, default=st.session_state.suggested_tools)
    extra_input = st.text_input("Extra instructions for the agent (optional)")
    if st.button("🚀 Run Cleaner"):
        with st.spinner("Agent cleaning in progress..."):
            cleaned, log = run_agent_pipeline(st.session_state.df, selected_tools, extra_input)
            st.session_state.cleaned_df = cleaned
            st.session_state.log = log
        st.success("✅ Cleaning complete.")

if st.session_state.cleaned_df is not None:
    st.subheader("📦 Final Cleaned Data")
    st.dataframe(st.session_state.cleaned_df)
    st.download_button("⬇ Download Cleaned CSV", st.session_state.cleaned_df.to_csv(index=False), "cleaned.csv", "text/csv")

    feedback = st.text_area("Still see issues? Describe them:")
    if st.button("🔁 Re-run With Feedback"):
        with st.spinner("Agent re-cleaning in progress..."):
            re_cleaned, re_log = run_agent_pipeline(st.session_state.cleaned_df, list(TOOLS.keys()), feedback)
            st.session_state.cleaned_df = re_cleaned
            st.session_state.log += re_log
        st.success("🔁 Re-cleaning complete.")
