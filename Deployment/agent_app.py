import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import ast
from typing import Dict, Any, List, Optional, TypedDict

from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# Patterns to detect common prompt injection attempts
SUSPICIOUS_PATTERNS = [
    r"(?i)ignore\s+previous\s+instructions",
    r"(?i)act\s+as\s+(an?\s+)?(admin|hacker|expert)",
    r"(?i)please\s+delete",
    r"(?i)execute\s+this\s+code",
    r"(?i)openai\.com|chatgpt|prompt injection",
]

def detect_prompt_injection(df: pd.DataFrame, sample_size: int = 500) -> List[str]:
    suspicious = []
    sample = df.astype(str).sample(min(len(df), sample_size), random_state=1)

    for col in sample.columns:
        for pattern in SUSPICIOUS_PATTERNS:
            matches = sample[col].str.contains(pattern, na=False, regex=True)
            if matches.any():
                suspicious.append(f"⚠️ Potential prompt injection pattern found in column '{col}'")
                break
    return suspicious

def validate_csv(df: pd.DataFrame) -> List[str]:
    issues = []
    if df.empty:
        issues.append("Uploaded CSV is empty.")
    if df.shape[1] < 2:
        issues.append("CSV should contain at least two columns.")
    if df.shape[0] < 3:
        issues.append("CSV should contain at least three rows.")
    if df.isnull().all(axis=1).any():
        issues.append("Some rows are completely empty.")
    return issues



# --- TOOL FUNCTIONS ---
#def fill_nulls_with_median(df): return df.fillna(df.median(numeric_only=True))
def normalize_missing_values(df): return df.replace(["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""], np.nan)
def drop_nulls(df):
    threshold = int(df.shape[1] * 0.5)  # Keep rows with at least 50% non-null values
    return df.dropna(thresh=threshold)
def standardize_column_names(df): df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_'); return df
def remove_duplicates(df): return df.drop_duplicates()
def convert_dtypes(df): return df.convert_dtypes()
def drop_columns_with_many_nulls(df, threshold: float = 0.8):
    """Drops columns where more than `threshold` proportion of values are null."""
    return df.loc[:, df.isnull().mean() <= threshold]
def standardize_booleans(df):
    bool_map = {"yes": True, "no": False, "1": True, "0": False}
    for col in df.columns:
        if df[col].astype(str).str.lower().isin(bool_map.keys()).any():
            df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(df[col])
    return df

TOOLS = {
    "drop_nulls": drop_nulls,
    #"fill_nulls_with_median": fill_nulls_with_median,
    "normalize_missing_values": normalize_missing_values,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "drop_columns_with_many_nulls": drop_columns_with_many_nulls,
    "standardize_booleans": standardize_booleans
}
def verify_tool_effect(before_df: pd.DataFrame, after_df: pd.DataFrame, tool_name: str) -> bool:
    """Check if applying the tool actually changed the DataFrame."""
    if tool_name == "fill_nulls_with_median":
        # Check if any numeric missing values were filled
        before_nulls = before_df.select_dtypes(include='number').isnull().sum().sum()
        after_nulls = after_df.select_dtypes(include='number').isnull().sum().sum()
        return after_nulls < before_nulls
    
    elif tool_name == "drop_nulls":
        return len(after_df) < len(before_df)

    elif tool_name == "drop_columns_with_many_nulls":
        return after_df.shape[1] < before_df.shape[1]

    elif tool_name == "remove_duplicates":
        return len(after_df) < len(before_df)

    elif tool_name == "standardize_column_names":
        return not before_df.columns.equals(after_df.columns)

    elif tool_name == "standardize_booleans":
        # Check if any boolean conversion happened
        before_obj_cols = before_df.select_dtypes(include='object')
        after_obj_cols = after_df.select_dtypes(include='object')
        return not before_obj_cols.equals(after_obj_cols)

    elif tool_name == "convert_dtypes":
        return any(before_df.dtypes != after_df.dtypes)

    elif tool_name == "normalize_missing_values":
        # Check if certain strings were converted to NaN
        return (before_df.replace(["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""], np.nan).isnull().sum().sum()
                != before_df.isnull().sum().sum())

    return True  # Default to true if unsure

# --- LLM Setup ---
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- Generate summary of columns ---
def generate_column_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    summary = []

    for col in df.columns:
        data = df[col]
        col_summary = {
            "Column": col,
            "Type": str(data.dtype),
            "Non-Null Count": data.notnull().sum(),
            "Missing Values": data.isnull().sum(),
            "Missing (%)": round(data.isnull().mean() * 100, 2),
            "Unique": data.nunique(),
            "Min": data.min() if pd.api.types.is_numeric_dtype(data) else "",
            "Max": data.max() if pd.api.types.is_numeric_dtype(data) else "",
            "Sample Values": ', '.join(map(str, data.dropna().unique()[:5])),
            "Num Outliers": "",  # default if not numeric
        }

        if pd.api.types.is_numeric_dtype(data):
            try:
                numeric_data = pd.to_numeric(data, errors="coerce")
                q1 = numeric_data.quantile(0.25)
                q3 = numeric_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                num_outliers = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).sum()
                col_summary["Num Outliers"] = int(num_outliers)
            except Exception as e:
                col_summary["Num Outliers"] = "Error"
                print(f"Error processing outliers for column {col}: {e}")

        summary.append(col_summary)

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
    feedback: str

def run_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    df = state["df"]
    tool_func = TOOLS.get(tool_name)

    if not tool_func:
        state["log"].append(f"❌ Tool '{tool_name}' not found.")
        return state

    try:
        before_df = df.copy(deep=True)
        after_df = tool_func(df.copy(deep=True))
        success = verify_tool_effect(before_df, after_df, tool_name)

        if success:
            state["df"] = after_df
            state["actions_taken"].append(tool_name)
            state["log"].append(f"✅ Ran tool: {tool_name}")
        else:
            state["log"].append(f"⚠️ Tool '{tool_name}' had no effect, skipping.")
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

def run_agent_pipeline(df: pd.DataFrame, allowed_tools: Optional[List[str]]=None, feedback: str = ""):
    if allowed_tools is None:
        allowed_tools=list(TOOLS.keys())
        
    def decider(state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        done = state["actions_taken"]
        step_count = state["step_count"]
        feedback = state.get("feedback", "")

        if step_count > 10:
            state["log"].append("⚠️ Max cleaning steps reached. Ending.")
            state["next_action"] = "end"
            return state

        instruction = f"""
You are a data cleaning agent. The dataset preview is:
{df.sample(20).to_string()}

Cleaning steps already applied: {done}
Allowed tools: {allowed_tools}
User provided these instructions (may include specific column issues):
\"\"\"{feedback}\"\"\"
Carefully check if they request a tool be applied again (even if used before).
If feedback mentions a column and a tool, prioritize applying that tool to that column.

Use this context to guide your next tool selection.
Respond ONLY with one of these tool names or 'end': {allowed_tools}
"""
        tool = llm.invoke(instruction).content.strip().strip("'\"").lower()

        if tool not in allowed_tools:
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
        "step_count": 0,
        "feedback": feedback
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
if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

if st.button("🧹 Reset"):
    for key in ["df", "suggested_tools", "cleaned_df", "feedback_history", "file_uploader"]:
        st.session_state[key] = None
    st.session_state.clear()
    st.rerun()

# --- Upload CSV ---
file = st.file_uploader("📂 Upload your CSV", type=["csv"], key="file_uploader")
if file:
    try:
        
        if file.size > 200 * 1024 * 1024:
            st.error("❌ File is too large (limit is 200MB).")
            st.stop()

        df = pd.read_csv(file)

        issues = validate_csv(df)
        issues += detect_prompt_injection(df)

        if issues:
            for issue in issues:
                st.error(f"❌ {issue}")
            st.stop()

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

    selected_tools = st.multiselect("Choose tools to apply", options=st.session_state.suggested_tools, default=st.session_state.suggested_tools)

    if st.button("🚀 Run Cleaner"):
        with st.spinner("Agent cleaning in progress..."):
            cleaned, log = run_agent_pipeline(st.session_state.df, selected_tools)
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
    if "log" in st.session_state:
        st.markdown("### 📝 Cleaning Log")
        for log_entry in st.session_state.log:
            st.write(log_entry)
            
        

    st.markdown("### 🗣️ Provide Feedback to Improve Cleaning")
    feedback = st.text_area("Still see issues? Describe them:")
    
    if st.button("🔁 Re-run With Feedback"):
        if feedback.strip():
            st.session_state.feedback_history.append(feedback.strip())
    
        combined_feedback = "\n".join(st.session_state.feedback_history)
    
        with st.spinner("Agent re-cleaning in progress..."):
            re_cleaned, re_log = run_agent_pipeline(
                st.session_state.cleaned_df,
                list(TOOLS.keys()),
                combined_feedback  
            )
            st.session_state.cleaned_df = re_cleaned
            st.session_state.log += re_log
        st.success("✅ Agent re-cleaning complete.")
        st.rerun()

