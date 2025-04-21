
import streamlit as st
import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, TypedDict, List, Optional
from langchain_core.runnables import RunnableConfig

### 1. TOOLS ###


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


tools = {
    "drop_nulls": drop_nulls,
    "fill_nulls_with_median": fill_nulls_with_median,
    "normalize_missing_values": normalize_missing_values,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "standardize_booleans": standardize_booleans,
    "generate_eda_summary": generate_eda_summary
}



### 2. LLM Decision Logic

# Pull API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

llm = ChatOpenAI(model="gpt-4", temperature=0)

def decide_next_action(state: Dict[str, Any]) -> str:
    df = state["df"]
    cleaned_steps = state["actions_taken"]

    instruction = f'''
You are a data cleaning agent. You have already taken these steps: {cleaned_steps}.
Here is a preview of the dataset (first 5 rows):
{df.head().to_string()}

Available tools: {list(tools.keys())}

Choose the **next most appropriate** cleaning tool to apply. 
If you're done, respond ONLY with 'end'.
Respond with just the tool name or 'end'.
'''

    response = llm.invoke(instruction).content.strip().lower()
    response = response.strip('"').strip("'")
    return response


### 3. Excecution node

def run_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    df = state["df"]
    tool_func = tools.get(tool_name)
    
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
    
### 4. Build LangGraph
def build_graph(llm_decision_func):
    builder = StateGraph(AgentState)

    builder.add_node("LLMDecision", llm_decision_func)

    # Tool runner node gets selected by LLM
    for tool_name in tools.keys():
        builder.add_node(tool_name, lambda state, t=tool_name: run_tool(state, t))

    builder.add_conditional_edges("LLMDecision", lambda x: x["next_action"], {
        **{name: name for name in tools.keys()},
        "end": END
    })

    # Loop back after each tool execution
    for tool_name in tools.keys():
        builder.add_edge(tool_name, "LLMDecision")

    builder.set_entry_point("LLMDecision")
    return builder.compile()



### Entrypoint
def run_agent_pipeline(df: pd.DataFrame):
    initial_state = {
        "df": df,
        "log": [],
        "actions_taken": [],
        "next_action": None,
    }

    def wrapper(state):
        action = decide_next_action(state)
        state["next_action"] = action
        return state

    # Pass the wrapper function here
    graph = build_graph(wrapper)

    final_state = graph.invoke(initial_state, config=RunnableConfig())
    return final_state["df"], final_state["log"]




### 6. Streamlit Frontend
st.set_page_config(page_title="Smart Data Cleaning Agent", layout="wide")
st.title("🧠 Smart Data Cleaning Agent")

# --- Session State Initialization ---
for key in ["df", "log", "cleaned_df", "step_selection"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "step_selection" else []

# --- Clear Session State ---
if st.button("🧹 Clear Session"):
    for key in ["df", "log", "cleaned_df", "step_selection"]:
        st.session_state[key] = None if key != "step_selection" else []
    st.rerun()

# --- File Upload ---
file = st.file_uploader("📂 Upload your CSV", type=["csv"])

if file:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    if df.empty:
        st.error("⚠️ Uploaded CSV is empty.")
    else:
        st.session_state.df = df
        st.write("📄 **Original Data**")
        st.dataframe(df, use_container_width=True)

        if st.button("🚀 Run Smart Cleaning Agent"):
            try:
                with st.spinner("Agent is working its cleaning magic..."):
                    cleaned_df, log = run_agent_pipeline(df)
                st.session_state.cleaned_df = cleaned_df
                st.session_state.log = log
                st.session_state.step_selection = [True if l.startswith("✅") else False for l in log]
                st.success("🎉 Cleaning complete! Review and adjust steps below.")
            except Exception as e:
                st.error(f"❌ Error during cleaning process: {e}")

# --- Step-by-Step Review and Toggle ---
if st.session_state.log:
    st.subheader("📝 Review & Control Cleaning Steps")
    st.caption("Uncheck any steps you want to exclude, then re-apply.")

    for i, step in enumerate(st.session_state.log):
        if step.startswith("✅ Ran tool: "):
            st.session_state.step_selection[i] = st.checkbox(
                label=step,
                value=st.session_state.step_selection[i],
                key=f"step_{i}"
            )
        else:
            st.markdown(step)

    if st.button("🔁 Apply Selected Steps"):
        try:
            selected_steps = [
                st.session_state.log[i].replace("✅ Ran tool: ", "").strip()
                for i, sel in enumerate(st.session_state.step_selection)
                if sel and st.session_state.log[i].startswith("✅")
            ]

            df_preview = st.session_state.df.copy()
            clean_log = []

            for tool in selected_steps:
                if tool in tools:
                    try:
                        df_preview = tools[tool](df_preview)
                        clean_log.append(f"✅ Re-applied tool: {tool}")
                    except Exception as e:
                        clean_log.append(f"❌ Error applying tool {tool}: {e}")

            st.session_state.cleaned_df = df_preview
            st.session_state.clean_log = clean_log
            st.success("Steps applied. See result below.")

        except Exception as e:
            st.error(f"❌ Error while applying steps: {e}")

# --- Final Output Preview and Download ---
if st.session_state.cleaned_df is not None:
    st.subheader("📦 Final Cleaned Data")
    st.dataframe(st.session_state.cleaned_df, use_container_width=True)

    if st.session_state.cleaned_df.empty:
        st.warning("⚠️ Your final result is empty. Try deselecting some steps.")
    else:
        st.download_button(
            label="⬇ Download Cleaned CSV",
            data=st.session_state.cleaned_df.to_csv(index=False),
            file_name="cleaned.csv",
            mime="text/csv"
        )
