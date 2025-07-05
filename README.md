# Installation
pip install graphiti-core
# Initialize Graphiti instance
graph = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_service="gemini"  
)

- Setup Langfuse:
`pip install langfuse openai`

- Run the Script and Analyze in Langfuse

Bash
python your_script_name.py
