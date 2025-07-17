import os
import json
import uuid
import time
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


class GraphMemory:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def log_tool_call(self, session_id, tool_name, args, result):
        call_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        with self.driver.session() as session:
            session.write_transaction(self._create_tool_memory,
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

    def check_memory(self):
        print("\n📚 Previous tool calls:")
        history = self.memory.get_tool_history(self.session_id)
        if not history:
            print("🔸 No memory yet.")
        for h in history:
            print(f"🔹 {h['timestamp']} | {h['tool_name']} | args={h['args']} → result={h['result']}")
        return history

    def call_tool(self, tool_name, args):
        print(f"🔧 Calling tool: {tool_name} with args: {args}")

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

    def respond_to_user(self, user_input):
        user_input = user_input.lower()

        if "buy" in user_input or "iphone" in user_input:
            self.check_memory()
            print("\n🛒 You asked to buy iPhone 15 Pro under 400 OMR")

            # Try red iPhone
            args1 = {"text": "iphone 15 pro", "filters": {"color": "red", "price_range": {"max": 400}}}
            result1 = self.call_tool("search_products", args1)

            if result1["total_found"] == 0:
                print("❌ Red color not found. Trying black...")
                args2 = {"text": "iphone 15 pro", "filters": {"color": "black", "price_range": {"max": 400}}}
                result2 = self.call_tool("search_products", args2)

                if result2["total_found"] > 0:
                    product = result2["products"][0]
                    print(f"✅ Found: {product['name']} in {product['color']} for {product['price']} OMR")
            else:
                print("✅ Found red iPhone!")

        elif "place order" in user_input:
            order_args = {"product_id": "p123", "user_id": "user001"}
            result = self.call_tool("place_order", order_args)
            if result["status"] == "success":
                print(f"📦 Order placed! Order number: {result['order_number']}")
            else:
                print("❌ Failed to place order.")

        elif "memory" in user_input:
            self.check_memory()

        elif user_input in ["exit", "quit"]:
            print("👋 Exiting chat.")
            return False

        else:
            print("🤖 I didn't understand. Try: 'buy iPhone', 'place order', or 'memory'")

        return True


# ─────────────────────────────────────────────────────────────
# Run Chat
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    memory = GraphMemory(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    session_id = "chat-001"
    agent = Agent(memory, session_id)

    print("🤖 Interactive Agent Chat (type 'exit' to quit)")
    while True:
        user_input = input("\n👤 You: ")
        if not agent.respond_to_user(user_input):
            break

    memory.close()
