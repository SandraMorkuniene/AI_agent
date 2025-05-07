import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import datetime
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
    return issues



# --- TOOL FUNCTIONS ---
def normalize_missing_values(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    targets = ["N/A", "n/a", "not available", "Not Available", "none", "None", "not a date", ""]
    if column:
        if column in df.columns:
            df[column] = df[column].replace(targets, np.nan)
    else:
        df = df.replace(targets, np.nan)
    return df


def standardize_column_names(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    # Column names are global; column arg ignored with a warning
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df


def remove_duplicates(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    # Works on full rows, column arg ignored
    return df.drop_duplicates()


def convert_dtypes(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    if column:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='ignore')
    else:
        df = df.convert_dtypes()
    return df


def remove_empty_rows(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    # Column arg irrelevant — we drop whole rows
    return df.dropna(how='all')


def drop_columns_with_80perc_nulls(df: pd.DataFrame, column: Optional[str] = None, threshold: float = 0.8) -> pd.DataFrame:
    # This tool is designed for full-column assessment; column arg ignored
    return df.loc[:, df.isnull().mean() <= threshold]


def standardize_booleans(df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
    bool_map = {"yes": True, "no": False, "Yes": True, "No": False, "1": True, "0": False}
    if column:
        if column in df.columns:
            df[column] = df[column].astype(str).str.lower().map(bool_map).fillna(df[column])
    else:
        for col in df.columns:
            if df[col].astype(str).str.lower().isin(bool_map.keys()).any():
                df[col] = df[col].astype(str).str.lower().map(bool_map).fillna(df[col])
    return df

TOOLS = {
    "remove_empty_rows": remove_empty_rows,
    "normalize_missing_values": normalize_missing_values,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "drop_columns_with_80perc_nulls": drop_columns_with_80perc_nulls,
    "standardize_booleans": standardize_booleans
}
def verify_tool_effect(before_df: pd.DataFrame, after_df: pd.DataFrame, tool_name: str) -> bool:
    """Check if applying the tool actually changed the DataFrame."""
    if tool_name == "remove_empty_rows":
        return len(after_df) < len(before_df)
        
    elif tool_name == "drop_columns_with_80perc_nulls":
        return after_df.shape[1] < before_df.shape[1]

    elif tool_name == "remove_duplicates":
        return len(after_df) < len(before_df)

    elif tool_name == "standardize_column_names":
        return not before_df.columns.equals(after_df.columns)

    elif tool_name == "standardize_booleans":
        before_obj_cols = before_df.select_dtypes(include='object')
        after_obj_cols = after_df.select_dtypes(include='object')
        return not before_obj_cols.equals(after_obj_cols)

    elif tool_name == "convert_dtypes":
        return any(before_df.dtypes != after_df.dtypes)

    elif tool_name == "normalize_missing_values":
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
    except Exception as e:
        print(f"Error parsing suggested tools: {e}")
        print(f"LLM response was: {response}")
        return []

# --- Agent State ---
class CleaningState(TypedDict):
    df: pd.DataFrame
    actions_taken: List[str]
    feedback: str 
    tool_decision: Optional[str] 
    column: Optional[str]
    available_tools: List[str]

# --- Tool Executor Node ---
def apply_tool(state: CleaningState) -> CleaningState:
    tool = state["tool_decision"]
    column = state["column"]
    df = state["df"]

    if tool not in TOOLS:
        return state

    try:
        before_df = df.copy()
        new_df = TOOLS[tool](before_df.copy(), column)

        changed = verify_tool_effect(before_df, new_df, tool)
        if changed:
            state["df"] = new_df
            log_entry = f"{tool}({column})" if column else tool
        else:
            log_entry = f"{tool}({column}) - no effect" if column else f"{tool} - no effect"

        state["actions_taken"].append(log_entry)
    except Exception as e:
        state["actions_taken"].append(f"Failed: {tool}({column}) -> {str(e)}")

    return state

# --- Tool Decision Node ---
def choose_tool(state: CleaningState) -> CleaningState:
    sample = state["df"].sample(min(20, len(state["df"])))
    prompt = f"""
You are a data cleaning assistant.

## Dataset Sample
{sample.to_string(index=False)}

## Cleaning History
{state["actions_taken"]}

## User Feedback
{state["feedback"]}

## Available Tools
{state.get("available_tools", list(TOOLS.keys()))}

Choose the next best tool to apply. If relevant, suggest a specific column too.
Respond in JSON format like:
{{ "tool": "normalize_missing_values", "column": "age" }}

If no further cleaning is needed, respond with:
{{ "tool": "end" }}
"""
    try:
        response = llm.invoke(prompt).content.strip()
        decision = json.loads(response)
        state["tool_decision"] = decision.get("tool")
        state["column"] = decision.get("column")
    except Exception as e:
        print(f"Error parsing tool decision: {e}")
        print(f"LLM response was: {response}")
        state["tool_decision"] = "end"
    return state

# --- Build LangGraph ---
workflow = StateGraph(CleaningState)
workflow.add_node("choose_tool", choose_tool)
workflow.add_node("apply_tool", apply_tool)
workflow.add_node("end", lambda s: s)

workflow.set_entry_point("choose_tool")
workflow.add_edge("choose_tool", "apply_tool")
workflow.add_conditional_edges(
    "apply_tool",
    lambda s: END if s["tool_decision"] == "end" else "choose_tool"
)

graph = workflow.compile()

# --- Run the agent ---
def run_agent_pipeline(df: pd.DataFrame, tools: List[str] = None, feedback: str = ""):
    initial_state = CleaningState(
        df=df,
        actions_taken=[],
        feedback=feedback,
        available_tools=tools or list(TOOLS.keys()),
        tool_decision=None,
        column=None
    )
    final_state = graph.invoke(initial_state)
    return final_state["df"], final_state["actions_taken"]




# --- Streamlit UI ---
st.set_page_config(page_title="Interactive Data Cleaner", layout="wide")
st.title("🧠 Interactive Data Cleaner Agent")

for key in ["df", "suggested_tools", "cleaned_df", "feedback_history", "log"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "feedback_history" else []

if st.button("🧹 Reset"):
    for key in ["df", "suggested_tools", "cleaned_df", "feedback_history", "file_uploader", "log"]:
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
if st.session_state["df"] is not None:
    if st.button("🧠 Analyze & Suggest Cleaning Steps"):
        st.session_state["suggested_tools"] = suggest_fixes(st.session_state["df"])

if st.session_state["suggested_tools"]:
    st.subheader("🔧 Suggested Cleaning Steps")
    all_tool_options = list(TOOLS.keys())
    tool_set = set(all_tool_options) | set(st.session_state["suggested_tools"])
    combined_tool_options = sorted(tool_set)
    valid_suggested_tools = [
        tool for tool in st.session_state["suggested_tools"] if tool in combined_tool_options
    ]

    selected_tools = st.multiselect(
        "Review, remove, or add tools below before running:",
        options=combined_tool_options,
        default=valid_suggested_tools,
        help="Only selected tools will be run. You can add or remove freely."
    )

    if st.button("🚀 Run Cleaner"):
        with st.spinner("Agent cleaning in progress..."):
            cleaned, log = run_agent_pipeline(st.session_state["df"], tools=selected_tools)
            st.session_state["cleaned_df"] = cleaned
            st.session_state["log"] = log
        st.success("✅ Cleaning complete.")

# --- Show Cleaned Data ---
if st.session_state["cleaned_df"] is not None:
    st.subheader("📦 Final Cleaned Data")
    st.dataframe(st.session_state["cleaned_df"].head(100), use_container_width=True)

    st.markdown("### 📊 Updated Column Summary")
    summary_df = generate_column_summary_table(st.session_state["cleaned_df"])
    st.dataframe(summary_df, use_container_width=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cleaned_{timestamp}.csv"
    st.download_button(
        "⬇ Download Cleaned CSV",
        st.session_state["cleaned_df"].to_csv(index=False),
        file_name=filename,
        mime="text/csv"
    )
    
    if st.session_state["log"]:
        st.markdown("### 📝 Cleaning Log")
        for log_entry in st.session_state["log"]:
            st.write(log_entry)
    else:
        st.write("No logs available.")
            
        

    st.markdown("### 🗣️ Provide Feedback to Improve Cleaning")
    feedback = st.text_area("Still see issues? Describe them:")

    if st.button("🔁 Re-run With Feedback"):
        if feedback.strip():
            st.session_state["feedback_history"].append(feedback.strip())

        combined_feedback = "\n".join(st.session_state["feedback_history"])

        with st.spinner("Agent re-cleaning in progress..."):
            re_cleaned, re_log = run_agent_pipeline(
                st.session_state["cleaned_df"],
                tools=selected_tools,
                feedback=combined_feedback)
            
            st.session_state["cleaned_df"] = re_cleaned
            st.session_state["log"] += re_log
        st.success("✅ Agent re-cleaning complete.")
        st.rerun()

