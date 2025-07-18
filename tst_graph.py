import os
import json
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from neo4j import GraphDatabase

load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function declarations (same as your original)
search_products_tool = genai.types.FunctionDeclaration(
    name="search_products",
    description="Searches the product catalog for items based on a text query and optional filters.",
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "text": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="A search query for products, like 'blue iphone' or 'samsung tv'."
            )
        },
        required=["text"]
    )
)

search_faqs_tool = genai.types.FunctionDeclaration(
    name="search_faqs",
    description="Searches the knowledge base for answers to frequently asked questions.",
    parameters=genai.types.Schema(
        type=genai.types.Type.OBJECT,
        properties={
            "text": genai.types.Schema(
                type=genai.types.Type.STRING,
                description="The user's question about shipping, returns, warranty, etc."
            )
        },
        required=["text"]
    )
)

TOOLS = [search_products_tool, search_faqs_tool]


class GraphMemory:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.enabled = True
            print("✅ Graph memory connected")
        except Exception as e:
            print(f"⚠️  Graph memory disabled: {e}")
            self.driver = None
            self.enabled = False

    def close(self):
        if self.driver:
            self.driver.close()

    def log_tool_call(self, session_id, tool_name, args, result):
        if not self.enabled:
            return

        call_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        try:
            with self.driver.session() as session:
                session.execute_write(self._create_tool_memory,
                                      session_id, call_id, timestamp,
                                      tool_name, args, result)
        except Exception as e:
            print(f"⚠️  Memory logging failed: {e}")

    @staticmethod
    def _create_tool_memory(tx, session_id, call_id, timestamp, tool_name, args, result):
        query = """
        MERGE (s:Session {id: $session_id})
        CREATE (c:ToolCall {
            id: $call_id,
            name: $tool_name,
            args: $args,
            timestamp: $timestamp
        })
        CREATE (r:ToolResponse {
            id: $call_id,
            result: $result,
            timestamp: $timestamp
        })
        MERGE (s)-[:HAS_CALL]->(c)
        MERGE (c)-[:GOT_RESULT]->(r)
        """
        tx.run(query, session_id=session_id, call_id=call_id,
               timestamp=timestamp, tool_name=tool_name,
               args=json.dumps(args), result=json.dumps(result))

    def get_tool_history(self, session_id, limit=5):
        if not self.enabled:
            return []

        query = """
        MATCH (s:Session {id: $session_id})-[:HAS_CALL]->(c:ToolCall)-[:GOT_RESULT]->(r:ToolResponse)
        RETURN c.name AS tool_name, c.args AS args, r.result AS result, c.timestamp AS timestamp
        ORDER BY c.timestamp DESC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, session_id=session_id, limit=limit)
                return [record.data() for record in result]
        except Exception as e:
            print(f"️  Memory retrieval failed: {e}")
            return []


class SimpleAgent:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash", tools=TOOLS)
        self.memory = GraphMemory(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USERNAME", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password123")
        )
        self.session_id = str(uuid.uuid4())

    def mock_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock function calls for testing"""
        if tool_name == "search_products":
            return {
                "products": [
                    {"name": "iPhone 15", "price": "$999", "color": args.get("text", "")},
                    {"name": "Samsung Galaxy S24", "price": "$899", "color": "blue"}
                ],
                "total_results": 2
            }
        elif tool_name == "search_faqs":
            return {
                "answer": "Shipping typically takes 3-5 business days for standard delivery.",
                "source": "FAQ Database"
            }
        return {"error": "Unknown function"}

    def chat(self, user_input: str) -> str:
        # Get memory context
        history = self.memory.get_tool_history(self.session_id)
        memory_context = ""
        if history:
            memory_context = "\n\nRecent interactions:\n"
            for h in reversed(history[-3:]):
                memory_context += f"- {h['tool_name']}: {h['args']}\n"


        prompt = f"""
User query: {user_input}
{memory_context}

Please help the user with their request. You can search products or FAQs if needed.
"""

        try:
            response = self.model.generate_content(prompt)

            if response.candidates and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]

                if part.function_call:

                    func_name = part.function_call.name
                    func_args = dict(part.function_call.args)

                    print(f" Calling {func_name} with {func_args}")


                    result = self.mock_tool_call(func_name, func_args)


                    self.memory.log_tool_call(self.session_id, func_name, func_args, result)


                    final_prompt = f"""
The user asked: {user_input}

I called {func_name} with {func_args} and got this result:
{json.dumps(result, indent=2)}

Please provide a helpful response to the user based on this information.
"""
                    final_response = self.model.generate_content(final_prompt)
                    return final_response.text if final_response else "Sorry, couldn't generate response."

                else:

                    return response.text

            return "Sorry, I couldn't understand that."

        except Exception as e:
            return f"Error: {e}"


def main():
    agent = SimpleAgent()
    print(" Simple Agent Ready! (Type 'quit' to exit)")
    print(f" Session ID: {agent.session_id}")

    try:
        while True:
            user_input = input("\n You: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break

            if user_input:
                response = agent.chat(user_input)
                print(f"🤖 Agent: {response}")

    except KeyboardInterrupt:
        print("\n Goodbye!")
    finally:
        agent.memory.close()


if __name__ == "__main__":
    main()
