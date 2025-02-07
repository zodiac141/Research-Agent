# AI Research Agent

## Overview

**AI Research Agent** is an advanced research assistant built using Streamlit. It leverages state-of-the-art language models via the Ollama interface (using the Deepseek model) to generate expert-level research reports. The agent can retrieve web resources, analyze user-uploaded documents (PDFs, CSVs, or text files), and combine both sources to produce detailed analyses, summaries, and actionable insights—all presented in an intuitive web interface.

## Features

- **Web Search Integration:**  
  Retrieve up to 10 relevant resources from the web based on the user query.

- **Document Analysis:**  
  Upload and process PDFs, CSVs, or TXT files to extract text and data for deeper analysis.

- **Advanced Summarization:**  
  Generate concise and insightful summaries for each source using custom prompt templates.

- **Expert-Level Reporting:**  
  Combine analysis from web and uploaded content to generate a comprehensive research report that mimics the output of a seasoned academic or Ph.D.-level expert.

- **Interactive UI:**  
  A user-friendly interface built with Streamlit for easy interaction and real-time results.

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [langchain_core](https://pypi.org/project/langchain_core/)
- [langchain_community](https://pypi.org/project/langchain_community/)
- [langchain_ollama](https://pypi.org/project/langchain_ollama/)
- [tavily-python](https://pypi.org/project/tavily-python/)
- [langgraph](https://pypi.org/project/langgraph/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [pandas](https://pypi.org/project/pandas/)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-research-agent.git
cd ai-research-agent
2. Set Up a Virtual Environment
Create and activate a virtual environment:

On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

On Windows:
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
Ensure your requirements.txt includes:

nginx
Copy
Edit
streamlit
langchain_core
langchain_community
langchain_ollama
tavily-python
langgraph
PyPDF2
pandas
Then, install them with:
pip install -r requirements.txt

4. Configure Environment Variables
Set the required API key for Tavily. For example:

On macOS/Linux:
export TAVILY_API_KEY="api key"

On Windows (PowerShell):
$env:TAVILY_API_KEY="api key"

5. Set Up Deepseek Model (via Ollama)
Ensure you have Ollama installed and pull the Deepseek model:

ollama pull deepseek-r1:7b

Usage
1. Run the Application
Start the Streamlit app:
streamlit run ai_research.py

If the streamlit command isn’t recognized, try:
python -m streamlit run ai_research.py

2. Interact with the Agent
Enter a Query:
Input your research question in the text box.

Upload Files:
Use the file uploader to attach PDFs, CSVs, or TXT files for deeper content analysis.

View the Report:
The agent will display a comprehensive research report including:

An expert-level analysis report.
Aggregated metrics and insights.
A bullet-point summary.
A list of sources used.