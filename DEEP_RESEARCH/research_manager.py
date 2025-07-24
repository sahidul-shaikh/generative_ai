from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from search_agent import search_agent
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from clarify_agent import clarify_agent
from agents import Runner, trace, gen_trace_id
import asyncio

class ResearchManager:

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and final report."""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting research...")
            search_plan = await self.plan_searches(query)
            yield "Searches planned, starting to search..."
            search_results = await self.perform_searches(search_plan)
            yield "Searches completed, writing report..."
            report = await self.write_report(query, search_results)
            yield "Report written, sending email..."
            await self.write_file(report)
            yield "Email save to file, research complete."
            yield report.markdown_report


    async def plan_searches(self, query: str) -> WebSearchPlan:
        """ Use the planner_agent to plan which searches to run for the query"""
        print("Planning searches ...")
        result = await Runner.run(planner_agent, f"Query: {query}")
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """call search() for each item in the search plan."""
        print("Searching...")
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = await asyncio.gather(*tasks)
        print("Finished searching...")
        return results
        
    async def search(self, item: WebSearchItem) -> str | None:
        """ Use the search agent to run a web search for each item in the search plan """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        result = await Runner.run(search_agent, input)
        return result.final_output
    
    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Use the writer agent to write a report based on the search results"""
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(writer_agent, input)
        print("Finished writing report")
        return result.final_output

    async def write_file(self, report: ReportData) -> None:
        """ Use the email_agent to write the email into a html file in local drive."""
        print("Writing email to file...")
        result = await Runner.run(email_agent, report.markdown_report)
        print("File write completed")
        return report
    
    async def generate_clarification_questions(self, query: str) -> str:
        """ Generate clarification questions based on the user's query """
        print("Generating clarification questions...")
        input = f"Please analyze this research query and generate 1 clarifying question that would help focus the research: {query}"
        try: 
            questions = await Runner.run(
                clarify_agent, 
                input
            )
            print("Generated clarification questions")
            return questions.final_output
        except Exception as e:
            print(f"Error generating questions: {e}")
            return ""
    