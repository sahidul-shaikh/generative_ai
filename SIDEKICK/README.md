# 🧠 Sidekick — Your Personal AI Co-Worker

**Sidekick** is an interactive AI assistant that can perform complex tasks using tools, evaluate its own performance, and ask for clarification when needed. Built with [LangGraph](https://docs.langchain.com/langgraph/), [LangChain](https://www.langchain.com/), [OpenAI](https://platform.openai.com/), and a [Gradio UI](https://www.gradio.app/), Sidekick brings reasoning, tool use, and feedback-based iteration into one cohesive workflow.

---

## Project Structure
sidekick/

│
├── app.py              # Gradio UI for user interaction

├── sidekick.py         # Core class with LangGraph implementation

├── sidekick_tools.py   # Tool definitions (search, file ops, wiki, Python REPL, push)

├── .env                # Environment variables (not checked in)

├── README.md           # You are here!

# Features
- Iterative assistant that refines its output using feedback
- Auto-evaluates responses based on success criteria
- Uses GPT-4o-mini via LangChain's ChatOpenAI

# Built-in tools:

 - Google Search (Serper API)
 - Wikipedia lookup
 - Python REPL for logic/math
 - File management (read/write/list/delete)
 - Push notifications (via Pushover)

# How It Works
1. You ask a question or give a task.
2. Sidekick uses tools if needed and gives an answer.
3. An evaluator LLM assesses the output based on success criteria.
4. If the answer is not good enough, it tries again or asks you for clarification.
5. Everything runs in a stateful LangGraph.

## Setup Instructions
1. **Clone the repository**
git clone [https://github.com/yourusername/sidekick.git](https://github.com/sahidul-shaikh/generative_ai.git)
cd generative_ai/sidekick

2. **Create a .env file**
OPENAI_API_KEY=your_openai_key
PUSHOVER_TOKEN=your_pushover_app_token
PUSHOVER_USER=your_pushover_user_key

3. **Install dependencies**
pip install -r requirements.txt
If you don’t have a requirements.txt yet, create one using:
pip freeze > requirements.txt

## Run the App
python app.py
This will launch a local Gradio app in your browser.

# Example Use Cases
- "Summarize this Wikipedia article."
- "Find the top 3 news headlines about climate change."
- "What is 2 to the power 12?"
- "Read this file and send me a push notification if it contains the word 'error'."

# Architecture Overview
flowchart TD

    UI[User Input via Gradio] -->|Message + Criteria| Worker
    
    Worker -->|Tool Calls| ToolNode
    
    ToolNode --> Worker
    
    Worker --> Evaluator
    
    Evaluator -->|Feedback| Worker
    
    Evaluator -->|Success/User Input Needed| END[End or Clarify]

# Future Improvements
- Memory persistence across sessions
- Enhanced tool selection via LangChain Toolkits
- Support for file uploads in the Gradio UI

