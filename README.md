# Installation
pip install graphiti-core
# Initialize Graphiti instance
graph = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_service="gemini"  
)
