# Installation
pip install graphiti-core
# Initialize Graphiti instance
graph = Graphiti(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_service="gemini"  
)

- Setup Langfuse with openai:
`pip install langfuse openai`

- Setup Langfuse with Gemini:
`pip install langfuse google-generativeai`

- Run the Script and Analyze in Langfuse

`python your_script_name.py`
