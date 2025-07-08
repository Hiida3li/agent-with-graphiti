async def debug_graphiti_embedding_flow():
    # Check Graphiti's embedder integration
    print(f"Graphiti embedder: {graphiti.embedder}")
    print(f"Embedder is same object: {graphiti.embedder is gemini_embedder}")

    # Check if there are recent facts with this issue
    recent_results = await graphiti.search(query="user preferences", num_results=5)

    for i, result in enumerate(recent_results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Fact: {result.fact}")
        print(f"Main fact_embedding is None: {result.fact_embedding is None}")

        if result.attributes and 'fact_embedding' in result.attributes:
            attr_embedding = result.attributes['fact_embedding']
            print(f"Attributes embedding length: {len(attr_embedding) if attr_embedding else 'None'}")
            print(f"Attributes embedding type: {type(attr_embedding)}")
            if attr_embedding and len(attr_embedding) > 0:
                print(f"First 3 values: {attr_embedding[:3]}")


await debug_graphiti_embedding_flow()
