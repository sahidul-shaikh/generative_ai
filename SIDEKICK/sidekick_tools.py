from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from typing import List, Any, Optional
import asyncio

load_dotenv(override=True)

# Push notification tool
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"


def tools() -> List[Any]:
    push_tool = Tool(
        name="send_push_notification",
        func=push,
        description="Use this tool when you want to send a push notification"
    )

    # Google serper tool
    serper = GoogleSerperAPIWrapper()
    tool_search =Tool(
            name="search",
            func=serper.run,
            description="Use this tool when you want to get the results of an online web search"
    )

    # File tool
    toolkit = FileManagementToolkit(root_dir="savedfile")
    file_tools = toolkit.get_tools()

    # Wiki tool
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    # Python tool
    python_repl = PythonREPLTool()

    return file_tools + [push_tool, tool_search, wiki_tool, python_repl]
