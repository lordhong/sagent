from langchain_community.tools.tavily_search import TavilySearchResults
# from stripe_agent_toolkit.crewai.toolkit import StripeAgentToolkit

tools = [TavilySearchResults(max_results=1)]