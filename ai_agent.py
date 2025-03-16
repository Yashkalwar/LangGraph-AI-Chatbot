# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup API Keys for Groq, OpenAI and Tavily
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

#Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    try:
        # Create LLM instance with API key when needed
        if provider == "Groq":
            if not GROQ_API_KEY:
                return {"error": "GROQ_API_KEY not found in environment"}
            llm = ChatGroq(model_name=llm_id, api_key=GROQ_API_KEY)
        elif provider == "OpenAI":
            if not OPENAI_API_KEY:
                return {"error": "OPENAI_API_KEY not found in environment"}
            llm = ChatOpenAI(model_name=llm_id, api_key=OPENAI_API_KEY)

        # Create search tool if allowed
        tools = []
        if allow_search:
            if not TAVILY_API_KEY:
                return {"error": "TAVILY_API_KEY not found in environment"}
            tools = [TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)]

        agent = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt
        )
        
        state = {"messages": query[0] if isinstance(query, list) else query}
        response = agent.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
        return {"response": ai_messages[-1] if ai_messages else "No response generated"}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
