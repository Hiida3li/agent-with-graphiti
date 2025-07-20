# import os
# import json
# import uuid
# import time
# from datetime import datetime
# from neo4j import GraphDatabase
# from dotenv import load_dotenv
# import google.generativeai as genai
#
#
# load_dotenv()
# NEO4J_URI = os.getenv("NEO4J_URI")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#
#
# genai.configure(api_key=GEMINI_API_KEY)
# gemini_model = genai.GenerativeModel("gemini-2.5-flash")
#
#
# class GraphMemory:
#     def __init__(self, uri, user, password):
#         self.driver = GraphDatabase.driver(uri, auth=(user, password))
#
#     def close(self):
#         self.driver.close()
#
#     def log_tool_call(self, session_id, tool_name, args, result):
#         call_id = str(uuid.uuid4())
#         timestamp = datetime.utcnow().isoformat()
#         with self.driver.session() as session:
#             session.execute_write(self._create_tool_memory,
#                                   session_id, call_id, timestamp,
#                                   tool_name, args, result)
#
#     @staticmethod
#     def _create_tool_memory(tx, session_id, call_id, timestamp, tool_name, args, result):
#         query = """
#         MERGE (s:Session {id: $session_id})
#         CREATE (c:ToolCall {
#             id: $call_id,
#             name: $tool_name,
#             args: $args,
#             timestamp: $timestamp
#         })
#         CREATE (r:ToolResponse {
#             id: $call_id,
#             result: $result,
#             timestamp: $timestamp
#         })
#         MERGE (s)-[:HAS_CALL]->(c)
#         MERGE (c)-[:GOT_RESULT]->(r)
#         """
#         tx.run(query, session_id=session_id, call_id=call_id,
#                timestamp=timestamp, tool_name=tool_name,
#                args=json.dumps(args), result=json.dumps(result))
#
#     def get_tool_history(self, session_id):
#         query = """
#         MATCH (s:Session {id: $session_id})-[:HAS_CALL]->(c:ToolCall)-[:GOT_RESULT]->(r:ToolResponse)
#         RETURN c.name AS tool_name, c.args AS args, r.result AS result, c.timestamp AS timestamp
#         ORDER BY c.timestamp ASC
#         """
#         with self.driver.session() as session:
#             result = session.run(query, session_id=session_id)
#             return [record.data() for record in result]
#
#
# class Agent:
#     def __init__(self, memory: GraphMemory, session_id: str):
#         self.memory = memory
#         self.session_id = session_id
#
#     def call_tool(self, tool_name, args):
#         print(f" Calling tool: {tool_name} with args: {args}")
#
#         if tool_name == "search_products":
#             color = args["filters"].get("color", "black")
#             if color == "red":
#                 result = {"total_found": 0, "products": []}
#             else:
#                 result = {
#                     "total_found": 1,
#                     "products": [
#                         {"id": "p123", "name": "iPhone 15 Pro", "color": color, "price": 395}
#                     ]
#                 }
#         elif tool_name == "place_order":
#             result = {"status": "success", "order_number": f"ORD-{uuid.uuid4().hex[:6]}"}
#         else:
#             result = {"status": "unknown tool"}
#
#         self.memory.log_tool_call(self.session_id, tool_name, args, result)
#         return result
#
#     def chat_with_gemini(self, user_input):
#         memory_trace = self.memory.get_tool_history(self.session_id)
#         if not memory_trace:
#             context = "No previous tool calls."
#         else:
#             context = "\n".join([
#                 f"{r['timestamp']} | {r['tool_name']} | args: {r['args']} → result: {r['result']}"
#                 for r in memory_trace
#             ])
#
#         prompt = f"""
# You are a helpful eCommerce customer service agent. Your job is to understand the user's intent and help them find the product information they need.
# If the tool result does not return any products that match the user's request:
# - Try alternative arguments or variants.
# - For example, if the user asked for a red product and nothing was found, retry with a different color.
# - If the user requested a product under 200 OMR and no result was found, retry with a slightly higher price (e.g., up to 250 OMR).
# Your goal is to find and recommend a suitable product, even if it means adjusting the filters intelligently to meet the user's needs as closely as possible.
#
# Memory:
# {context}
#
# User said: "{user_input}"
#
# Reply helpfully using the memory context.
# """
#         response = gemini_model.generate_content(prompt)
#         return response.text.strip()
#
#     def respond_to_user(self, user_input):
#         if user_input.lower() in ["exit", "quit"]:
#             print(" Exiting chat.")
#             return False
#
#         reply = self.chat_with_gemini(user_input)
#         print(f" {reply}")
#         return True
#
#
# if __name__ == "__main__":
#     memory = GraphMemory(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
#     session_id = "chat-001"
#     agent = Agent(memory, session_id)
#
#     print(" Interactive Agent Chat with Gemini (type 'exit' to quit)")
#     while True:
#         user_input = input("\n You: ")
#         if not agent.respond_to_user(user_input):
#             break
#
#     memory.close()


import os
import json
import uuid
from datetime import datetime, timezone
from neo4j import GraphDatabase
from dotenv import load_dotenv
from google import genai

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Create the centralized client
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define tools for function-calling feature
search_products_tool = {
    "name": "search_products",
    "description": "Searches the product catalog for items based on a text query and optional filters.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A search query for products."
            },
            "filters": {
                "type": "object",
                "description": "Optional filters like color.",
                "properties": {
                    "color": {"type": "string"}
                }
            }
        },
        "required": ["query"]
    }
}

place_order_tool = {
    "name": "place_order",
    "description": "Places an order for a given product ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "The unique ID of the product to order."
            },
            "quantity": {
                "type": "integer",
                "description": "The number of items to order."
            }
        },
        "required": ["product_id", "quantity"]
    }
}


class GraphMemory:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def log_tool_call(self, session_id, tool_name, args, result):
        call_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.driver.session() as session:
            session.execute_write(self._create_tool_memory,
                                  session_id, call_id, timestamp,
                                  tool_name, args, result)

    @staticmethod
    def _create_tool_memory(tx, session_id, call_id, timestamp, tool_name, args, result):
        query = """
        MERGE (s:Session {id: $session_id})
        CREATE (c:ToolCall {
            id: $call_id,
            name: $tool_name,
            args: $args,
            result: $result,
            timestamp: $timestamp
        })
        MERGE (s)-[:HAS_CALL]->(c)
        """
        tx.run(query, session_id=session_id, call_id=call_id,
               timestamp=timestamp, tool_name=tool_name,
               args=json.dumps(args), result=json.dumps(result))

    def get_tool_history(self, session_id):
        query = """
        MATCH (s:Session {id: $session_id})-[:HAS_CALL]->(c:ToolCall)
        RETURN c.name AS tool_name, c.args AS args, c.result AS result, c.timestamp AS timestamp
        ORDER BY c.timestamp ASC
        """
        with self.driver.session() as session:
            result = session.run(query, session_id=session_id)
            return [
                {
                    "tool_name": record["tool_name"],
                    "args": json.loads(record["args"]),
                    "result": json.loads(record["result"]),
                    "timestamp": record["timestamp"]
                } for record in result
            ]

    def get_conversation_context(self, session_id):
        """Get formatted conversation context for the AI model"""
        history = self.get_tool_history(session_id)
        if not history:
            return "No previous tool calls."

        context_lines = []
        for record in history:
            context_lines.append(
                f"{record['timestamp']} | {record['tool_name']} | "
                f"args: {record['args']} → result: {record['result']}"
            )
        return "\n".join(context_lines)


class Agent:
    def __init__(self, memory: GraphMemory, session_id: str):
        self.memory = memory
        self.session_id = session_id
        self.client = client  # Use the global client
        self.tools = [search_products_tool, place_order_tool]

        # System prompt for the agent
        self.system_prompt = """
You are a helpful eCommerce customer service agent. Your job is to understand the user's intent and help them find the product information they need.

If the tool result does not return any products that match the user's request:
- Try alternative arguments or variants.
- For example, if the user asked for a red product and nothing was found, retry with a different color.
- If the user requested a product under 200 OMR and no result was found, retry with a slightly higher price (e.g., up to 250 OMR).

Your goal is to find and recommend a suitable product, even if it means adjusting the filters intelligently to meet the user's needs as closely as possible.
"""

    def respond_to_user(self, user_input: str) -> bool:
        """Handles a user's message and returns a response."""
        if user_input.lower() in ["exit", "quit"]:
            print("🤖 Exiting chat.")
            return False

        try:
            # Get conversation context from memory
            context = self.memory.get_conversation_context(self.session_id)

            # First, determine if we need to call a tool based on user input
            if self._should_search_products(user_input):
                # Extract search parameters from user input
                search_params = self._extract_search_params(user_input)
                print(f"🤖 Searching for products with: {search_params}")

                tool_result = self.call_tool("search_products", search_params)

                # Generate response based on tool result
                response_prompt = f"""
{self.system_prompt}

Previous conversation context:
{context}

User request: {user_input}
Search results: {tool_result}

Provide a helpful response about the search results. If no products were found, suggest alternatives.
"""

                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=response_prompt
                )
                print(f"🤖 {response.text.strip()}")

            elif self._should_place_order(user_input):
                # Extract order parameters
                order_params = self._extract_order_params(user_input)
                print(f"🤖 Placing order with: {order_params}")

                tool_result = self.call_tool("place_order", order_params)

                response_prompt = f"""
{self.system_prompt}

Previous conversation context:
{context}

User request: {user_input}
Order result: {tool_result}

Provide a confirmation message about the order.
"""

                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=response_prompt
                )
                print(f"🤖 {response.text.strip()}")

            else:
                # Direct conversation without tools
                conversation_prompt = f"""
{self.system_prompt}

Previous conversation context:
{context}

User: {user_input}

Respond helpfully. If the user is looking for products, let them know you can search for them.
"""

                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=conversation_prompt
                )
                print(f"🤖 {response.text.strip()}")

        except Exception as e:
            print(f"🤖 An error occurred: {e}")

        return True

    def _should_search_products(self, user_input: str) -> bool:
        """Determine if user input indicates a product search."""
        search_keywords = ["looking for", "search", "find", "phone", "product", "buy", "purchase", "show me"]
        return any(keyword in user_input.lower() for keyword in search_keywords)

    def _should_place_order(self, user_input: str) -> bool:
        """Determine if user input indicates placing an order."""
        order_keywords = ["order", "buy this", "purchase this", "place order", "checkout"]
        return any(keyword in user_input.lower() for keyword in order_keywords)

    def _extract_search_params(self, user_input: str) -> dict:
        """Extract search parameters from user input."""
        params = {"query": user_input}

        # Extract color filter
        colors = ["red", "blue", "green", "black", "white", "silver", "gold"]
        for color in colors:
            if color in user_input.lower():
                params["filters"] = {"color": color}
                break

        return params

    def _extract_order_params(self, user_input: str) -> dict:
        """Extract order parameters from user input."""
        # This is a simple implementation - in practice, you'd want more sophisticated parsing
        return {
            "product_id": "p123",  # Would extract from context/previous search
            "quantity": 1
        }

    def call_tool(self, tool_name: str, args: dict):
        """Executes a tool and logs the interaction to graph memory."""
        if tool_name == "search_products":
            color = args.get("filters", {}).get("color", "black")
            if color == "red":
                result = {"total_found": 0, "products": []}
            else:
                result = {
                    "total_found": 1,
                    "products": [
                        {"id": "p123", "name": "iPhone 15 Pro", "color": color, "price": 395}
                    ]
                }
        elif tool_name == "place_order":
            result = {"status": "success", "order_number": f"ORD-{uuid.uuid4().hex[:6]}"}
        else:
            result = {"status": "unknown tool"}

        # Log the tool call to graph memory
        self.memory.log_tool_call(self.session_id, tool_name, args, result)
        return result


if __name__ == "__main__":
    memory = GraphMemory(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    session_id = f"chat-{uuid.uuid4().hex[:8]}"  # Generate unique session ID
    agent = Agent(memory, session_id)

    print(" Interactive Agent Chat with Gemini (type 'exit' to quit)")
    print(f" Session ID: {session_id}")

    while True:
        user_input = input("\n👤 You: ")
        if not agent.respond_to_user(user_input):
            break

    memory.close()
    print(" Chat session ended. Memory saved to graph database.")


