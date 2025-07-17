import os
import json
import uuid
import time
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


class GraphMemory:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def log_tool_call(self, session_id, tool_name, args, result):
        call_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
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

    def get_tool_history(self, session_id):
        query = """
        MATCH (s:Session {id: $session_id})-[:HAS_CALL]->(c:ToolCall)-[:GOT_RESULT]->(r:ToolResponse)
        RETURN c.name AS tool_name, c.args AS args, r.result AS result, c.timestamp AS timestamp
        ORDER BY c.timestamp ASC
        """
        with self.driver.session() as session:
            result = session.run(query, session_id=session_id)
            return [record.data() for record in result]


class Agent:
    def __init__(self, memory: GraphMemory, session_id: str):
        self.memory = memory
        self.session_id = session_id

    def call_tool(self, tool_name, args):
        print(f" Calling tool: {tool_name} with args: {args}")

        # Dummy tool behavior
        if tool_name == "search_products":
            color = args["filters"].get("color", "black")
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

        self.memory.log_tool_call(self.session_id, tool_name, args, result)
        return result

    def chat_with_gemini(self, user_input):
        memory_trace = self.memory.get_tool_history(self.session_id)
        if not memory_trace:
            context = "No previous tool calls."
        else:
            context = "\n".join([
                f"{r['timestamp']} | {r['tool_name']} | args: {r['args']} → result: {r['result']}"
                for r in memory_trace
            ])

        prompt = f"""
You are a helpful and proactive eCommerce customer service agent. Your job is to understand the user's intent and help them find the product information they need.
If the tool result does not return any products that match the user's request:
- Try alternative arguments or variants.
- For example, if the user asked for a red product and nothing was found, retry with a different color.
- If the user requested a product under 200 OMR and no result was found, retry with a slightly higher price (e.g., up to 250 OMR).
Your goal is to find and recommend a suitable product, even if it means adjusting the filters intelligently to meet the user's needs as closely as possible.

Memory:
{context}

User said: "{user_input}"

Reply helpfully using the memory context.
"""
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

    def respond_to_user(self, user_input):
        if user_input.lower() in ["exit", "quit"]:
            print(" Exiting chat.")
            return False

        reply = self.chat_with_gemini(user_input)
        print(f" {reply}")
        return True


if __name__ == "__main__":
    memory = GraphMemory(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    session_id = "chat-001"
    agent = Agent(memory, session_id)

    print(" Interactive Agent Chat with Gemini (type 'exit' to quit)")
    while True:
        user_input = input("\n You: ")
        if not agent.respond_to_user(user_input):
            break

    memory.close()
