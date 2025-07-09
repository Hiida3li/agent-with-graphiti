import os
import datetime
import nest_asyncio
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.utils.bulk_utils import EpisodeType
nest_asyncio.apply()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD not set in .env")

async def simple_graphiti_gemini_example():
    """
    Simple example showing how to use Graphiti with Gemini
    """

    gemini_llm_client = GeminiClient(
        config=LLMConfig(
            api_key=GOOGLE_API_KEY,
            model="gemini-2.5-flash"
        )
    )

    gemini_embedder = GeminiEmbedder(
        config=GeminiEmbedderConfig(
            api_key=GOOGLE_API_KEY,
            embedding_model="text-embedding-004"
        )
    )

    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
        llm_client=gemini_llm_client,
        embedder=gemini_embedder
    )

    print(" Graphiti initialized with Gemini!")

    await graphiti.build_indices_and_constraints()
    print(" Database schema initialized!")

    episodes = [
        "user033 is Omani always buy Apple brand less than 500 OMR.",
        "user005 always ask about latest new products from Apple, and ordered 3 Apple smart watches.",
        "user001 like same brand as user033. He makes new order every friday.",
        "user005 recently looking for a new smart watches from Samsung and no more likes Apple brand.",
        "user001 preferences are iphone 15 pro, Apple Watch Ultra 2, and Airpods 3X pro"
    ]

    print("\n Adding episodes to knowledge graph...")
    for i, episode in enumerate(episodes, 1):
        await graphiti.add_episode(
            name=f"episode_{i}",
            episode_body=episode,
            source=EpisodeType.text,
            source_description="User preferences",
            reference_time=datetime.datetime.now()
        )
        print(f" Added episode {i}")

    print("\n Querying the knowledge graph...")

    queries = [
        "what's the preferred brand for user005?"
    ]

    for query in queries:
        results = await graphiti.search(query=query, num_results=3)
        print(f"\n Results for query: {query}")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result}")

if __name__ == "__main__":
    asyncio.run(simple_graphiti_gemini_example())

# await simple_graphiti_gemini_example()
