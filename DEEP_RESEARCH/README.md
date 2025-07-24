# 🧠 Deep Research App (Agentic AI Powered)

A modular, agentic AI-powered research assistant built using the OpenAI Agent SDK. This app guides users through the full research lifecycle: from refining their query, planning searches, summarizing findings, generating reports, and saving them in a well-formatted HTML email.

## 🚀 Features

- **Clarification Agent**: Refines vague queries by generating clarifying questions.
- **Planner Agent**: Plans the most relevant web search queries based on the original query.
- **Search Agent**: Conducts concise web research using OpenAI and returns summarized results.
- **Writer Agent**: Synthesizes findings into a cohesive, well-structured markdown report.
- **Email Agent**: Converts the report into HTML and saves it locally for sharing.
- **Gradio UI**: A user-friendly interface to interact with the system in real time.

## 🧩 Agent Architecture

This system is powered by multiple specialized agents:

| Agent            | Responsibility                                             |
|------------------|------------------------------------------------------------|
| `ClarifyAgent`   | Generates clarification questions to narrow research scope |
| `PlannerAgent`   | Plans what web searches are needed                         |
| `SearchAgent`    | Performs web search and summarizes findings                |
| `WriterAgent`    | Generates a full markdown report from search summaries     |
| `EmailAgent`     | Formats and saves the report as an HTML email              |

## 📁 Project Structure

├── deep_research.py # Entry point with Gradio UI

├── research_manager.py # Orchestrates the full agentic research pipeline

├── planner_agent.py # Creates the search plan

├── search_agent.py # Summarizes search results

├── writer_agent.py # Writes the final report

├── email_agent.py # Saves the report as HTML

├── clarify_agent.py # Generates clarifying questions

├── agents/ # Shared agent utilities (assumed provided externally)

│ ├── init.py # Agent class and Runner utilities

│ └── model_settings.py # Model configuration

├── mail.html # Output HTML report (generated)

├── .env # Environment config for OpenAI credentials

## 🛠️ Setup Instructions

### 1. Clone the repository

git clone [https://github.com/your-username/deep-research-agentic.git](https://github.com/sahidul-shaikh/generative_ai.git)

cd DEEP_RESEARCH

### 2. Install dependencies
Make sure you have Python 3.9+ and pip installed.

pip install -r requirements.txt
You may create a requirements.txt with:

gradio
python-dotenv
openai
pydantic

### 3. Set up environment variables
Create a .env file in the root directory with your OpenAI API key:

OPENAI_API_KEY=your_openai_key_here

### 4. Run the app

python deep_research.py
The Gradio UI will open in your default browser.

🧪 Example Workflow

1. Enter a topic like:
"Impacts of AI on cybersecurity in 2025"

2. Click "Generate Clarification Questions"

3. Answer the clarification prompts (optional, but helps refine the query)

4. Click "Do Research"

5. The app will:

    - Plan the best search terms

    - Fetch summaries from the web

    - Generate a long-form report

    - Save it as an HTML email (mail.html)

📬 Output

The report will be saved locally as mail.html

It includes:

- A subject

- Well-formatted HTML body with the complete research report

🧠 Built With

- OpenAI GPT-4o Agents (via Agent SDK)

- Gradio (for UI)

- Pydantic (for typed models)

- Asyncio (for concurrent searching)

- dotenv (for config management)

📌 Notes
- Make sure the agents folder includes your custom Agent, Runner, trace, function_tool, etc.

- Ensure you are using a model with access to tools (e.g., gpt-4o-mini with web search enabled).

