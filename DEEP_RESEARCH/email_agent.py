from agents import Agent, function_tool
from typing import Dict

@function_tool
def write_to_file(email_subject: str, email_body: str) -> Dict[str, str]:
    """Write email messaage to a html file in local drive"""
    with open('mail.html', 'w') as file:
        file.write(email_subject + "\n\n")
        file.write(email_body + "\n")
    return {"status":"success"}

INSTRUCTIONS = """You are able to save a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to save one email as a html file in the local drive, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

email_agent = Agent(
    name="Email Agent",
    instructions=INSTRUCTIONS,
    model="gpt-4o-mini",
    tools=[write_to_file]
)