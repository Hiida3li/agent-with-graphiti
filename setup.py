import os
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv

import google.generativeai as genai
from neo4j import AsyncGraphDatabase
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

load_dotenv()

# Set up Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Neo4j setup
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")
