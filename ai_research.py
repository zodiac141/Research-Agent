import streamlit as st
import re
from itertools import chain
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
import PyPDF2
import pandas as pd
import io

# -------------------- Enhanced Prompt Templates --------------------
advanced_analysis_template = """
You are a world-class research expert with a doctorate-level background in the subject.
Your task is to carefully analyze the following aggregated content collected from both online sources and uploaded documents.
Please perform the following:
1. Provide an overview including the number of documents and datasets.
2. Compute key metrics such as average word counts, data variability, and central themes.
3. Highlight novel insights, correlations, or actionable recommendations.
4. Synthesize your analysis into a scholarly, bullet-pointed report.

Query: {query}
Uploaded Content: {uploaded_content}
Web Content: {web_content}
Expert Report:
"""

advanced_summary_template = """
You are an expert summarizer. Generate a concise summary for the following content that:
- Directly addresses the query.
- Highlights critical findings and insights.
- Reflects the clarity and depth of a seasoned professor.

Query: {query}
Content: {content}
Summary:
"""

advanced_response_template = """
You are a highly experienced research analyst. Based on the query, the aggregated metrics and analysis from both web and uploaded content,
and the detailed summaries, generate a comprehensive response. Your answer should:
1. Directly and precisely answer the query.
2. Integrate key metrics and insights.
3. Provide actionable recommendations.
4. Be structured, clear, and scholarly.

Question: {question}
Aggregated Analysis: {metrics}
Summarized Content: {summaries}
Final Answer:
"""

# -------------------- Extended TypedDicts --------------------
class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    uploaded_content: str  # Text extracted from PDFs or CSVs
    summarized_results: list[str]
    metrics: str
    response: str

class ResearchStateInput(TypedDict):
    query: str

class ResearchStateOutput(TypedDict):
    sources: list[str]
    metrics: str
    response: str

# -------------------- Node Functions --------------------

def search_web(state: ResearchState):
    search = TavilySearchResults(max_results=10)
    search_results = search.invoke(state["query"])
    
    sources = []
    web_results = []
    
    for result in search_results:
        # Check if result is a dictionary
        if isinstance(result, dict):
            # Use .get to provide a default empty string if key is missing.
            sources.append(result.get('url', ''))
            web_results.append(result.get('content', ''))
        # If result is a string, then treat it as both the URL and the content.
        elif isinstance(result, str):
            sources.append(result)
            web_results.append(result)
        else:
            # Fallback: convert any other type to string
            sources.append(str(result))
            web_results.append(str(result))
    
    return {
        "sources": sources,
        "web_results": web_results
    }

def process_uploaded_files(state: ResearchState):
    """
    Process uploaded files (PDFs or CSVs) and extract text.
    Assumes that st.file_uploader was used in the main UI to store files in session_state.
    """
    uploaded_texts = []
    # 'uploaded_files' is assumed to be in the session_state as a list of UploadedFile objects
    if "uploaded_files" in st.session_state:
        for uploaded_file in st.session_state.uploaded_files:
            if uploaded_file.name.lower().endswith(".pdf"):
                # Extract text from PDF
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                uploaded_texts.append(text)
            elif uploaded_file.name.lower().endswith((".csv", ".txt")):
                # For CSV or TXT, read as text (for CSV, you might want to use pandas to summarize numerical data)
                content = uploaded_file.getvalue().decode("utf-8")
                # Optionally, if CSV, process with pandas
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(io.StringIO(content))
                    # Create a quick summary of the CSV (you can expand this as needed)
                    summary = f"CSV Summary: {df.describe(include='all').to_string()}"
                    uploaded_texts.append(summary)
                else:
                    uploaded_texts.append(content)
    return {"uploaded_content": "\n\n".join(uploaded_texts)}

def analyze_data(state: ResearchState):
    """Aggregate and analyze both web and uploaded content to produce expert-level metrics."""
    model = ChatOllama(model="deepseek-r1:7b")
    prompt = ChatPromptTemplate.from_template(advanced_analysis_template)
    chain = prompt | model

    # Combine web results and uploaded content
    web_content = "\n\n".join(state["web_results"])
    uploaded_content = state.get("uploaded_content", "")
    analysis_result = chain.invoke({
        "query": state["query"],
        "web_content": web_content,
        "uploaded_content": uploaded_content
    })
    metrics = clean_text(analysis_result.content)
    return {"metrics": metrics}

def summarize_results(state: ResearchState):
    """Summarize each piece of content (from web sources) using an advanced summarization prompt."""
    model = ChatOllama(model="deepseek-r1:7b")
    prompt = ChatPromptTemplate.from_template(advanced_summary_template)
    chain = prompt | model

    summarized_results = []
    for content in state["web_results"]:
        summary = chain.invoke({"query": state["query"], "content": content})
        summarized_results.append(clean_text(summary.content))
    return {"summarized_results": summarized_results}

def generate_response(state: ResearchState):
    """Generate the final, comprehensive expert response using the advanced response prompt."""
    model = ChatOllama(model="deepseek-r1:7b")
    prompt = ChatPromptTemplate.from_template(advanced_response_template)
    chain = prompt | model

    combined_summary = "\n\n".join(state["summarized_results"])
    return {
        "response": chain.invoke({
            "question": state["query"],
            "metrics": state["metrics"],
            "summaries": combined_summary
        })
    }

def clean_text(text: str):
    """Clean the text by removing any embedded <think> ... </think> segments."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

# -------------------- Build the Pipeline --------------------
builder = StateGraph(
    ResearchState,
    input=ResearchStateInput,
    output=ResearchStateOutput
)

builder.add_node("search_web", search_web)
builder.add_node("process_uploaded_files", process_uploaded_files)
builder.add_node("analyze_data", analyze_data)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_response)

# Pipeline Order:
# 1. Search the web
# 2. Process any uploaded files
# 3. Analyze aggregated content (both web and uploads)
# 4. Summarize web results
# 5. Generate final response
builder.add_edge(START, "search_web")
builder.add_edge("search_web", "process_uploaded_files")
builder.add_edge("process_uploaded_files", "analyze_data")
builder.add_edge("analyze_data", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

# -------------------- Streamlit Interface --------------------
st.title("Advanced AI Research Agent")

# Input query
query = st.text_input("Enter your research query:")

# File uploader for PDFs, CSVs, etc.
uploaded_files = st.file_uploader("Upload PDFs, CSVs, or TXT files for deeper analysis:", accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

if query:
    # Invoke the pipeline
    response_state = graph.invoke({"query": query})
    
    st.subheader("Final Expert Response:")
    st.write(clean_text(response_state["response"].content))
    
    st.subheader("Aggregated Analysis & Metrics:")
    st.write(response_state["metrics"])
    
    st.subheader("Sources:")
    for source in response_state["sources"]:
        st.write(source)
