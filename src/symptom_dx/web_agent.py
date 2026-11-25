"""Web search agent: the LLM decides what to search on DuckDuckGo and summarizes what it finds."""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate

from symptom_dx.prompts import WEB_AGENT_SYSTEM_PROMPT, question_for


class WebSearchRetriever:
    name = "Web search"

    def __init__(self, llm, verbose: bool = False):
        # A real region is needed: the default "wt-wt" breaks the Wikipedia fallback search.
        search = DuckDuckGoSearchAPIWrapper(region="us-en", max_results=5)
        tools = [DuckDuckGoSearchRun(api_wrapper=search)]
        prompt = ChatPromptTemplate.from_messages(
            [("system", WEB_AGENT_SYSTEM_PROMPT), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, max_iterations=4, verbose=verbose)

    def retrieve(self, symptoms: list[str]) -> str:
        return self.executor.invoke({"input": question_for(symptoms)})["output"]
