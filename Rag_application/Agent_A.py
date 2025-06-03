from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from pydantic import BaseModel, Field
from typing import List
from tavily import TavilyClient
from scrapegraph_py import Client
from langchain_ollama import ChatOllama

import os
import json

output_dir = "./ai-agent-output"
os.makedirs(output_dir, exist_ok=True)

llm = ChatOllama(model="llama3.2:1b")  
search_client = TavilyClient(api_key="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")
#scrape_client = Client(api_key=)

