
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from typing import Dict, Any
from langchain_core.runnables import RunnableConfig

### 1. TOOLS ###

@tool
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

@tool
def fill_nulls_with_median(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))

@tool
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

@tool
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

@tool
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return df.convert_dtypes()

@tool
def generate_eda_summary(df: pd.DataFrame) -> str:
    return df.describe().to_string()

tools = {
    "drop_nulls": drop_nulls,
    "fill_nulls_with_median": fill_nulls_with_median,
    "standardize_column_names": standardize_column_names,
    "remove_duplicates": remove_duplicates,
    "convert_dtypes": convert_dtypes,
    "generate_eda_summary": generate_eda_summary,
}



### 2. LLM Decision Logic
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
    return response


### 3. Excecution node

def run_tool(state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    df = state["df"]
    tool_func = tools.get(tool_name)
    if tool_func is None:
        state["log"].append(f"❌ Tool '{tool_name}' not found.")
        return state
    new_df = tool_func(df)
    state["df"] = new_df
    state["actions_taken"].append(tool_name)
    state["log"].append(f"✅ Ran tool: {tool_name}")
    return state



### 4. Build LangGraph
def build_graph():
    builder = StateGraph()

    builder.add_node("LLMDecision", decide_next_action)

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
    graph = build_graph()
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

    graph.get_node("LLMDecision").set_callable(wrapper)
    final_state = graph.invoke(initial_state, config=RunnableConfig())
    return final_state["df"], final_state["log"]




### 6. Streamlit Frontend
st.title("🧠 Smart Data Cleaning Agent")
file = st.file_uploader("Upload your CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.write("📄 Original Data")
    st.dataframe(df)

    if st.button("Run Smart Cleaning"):
        cleaned_df, log = run_agent_pipeline(df)
        st.write("✅ Cleaned Data")
        st.dataframe(cleaned_df)

        st.write("📝 Agent Log")
        for step in log:
            st.markdown(step)

        st.download_button("⬇ Download Cleaned Data", cleaned_df.to_csv(index=False), "cleaned.csv")



