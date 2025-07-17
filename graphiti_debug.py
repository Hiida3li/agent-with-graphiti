from neo4j import GraphDatabase
import uuid
from datetime import datetime
import time
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


# ─────────────────────────────────────────────────────────────
# Graph Memory Class
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Agent Class
# ─────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, memory: GraphMemory, session_id: str):
        self.memory = memory
        self.session_id = session_id

    def call_tool(self, tool_name, args):
        print(f"🔧 Calling tool: {tool_name} with args: {args}")

        # Simulated tool behavior
        if tool_name == "search_products":
            if args['filters']['color'] == "red":
                result = {"total_found": 0, "products": []}
            else:
                result = {"total_found": 1, "products": [{"name": "iPhone 15 Pro", "color": "black", "price": 395}]}
        elif tool_name == "place_order":
            result = {"status": "success", "order_number": "ORD9981"}
        else:
            result = {"status": "unknown tool"}

        # Log to memory
        self.memory.log_tool_call(self.session_id, tool_name, args, result)
        return result

    def handle_user_request(self):
        print("\n📚 Checking previous tool calls...")
        history = self.memory.get_tool_history(self.session_id)
        if history:
            for h in history:
                print(f"🔹 {h['timestamp']} | {h['tool_name']} | args={h['args']} → result={h['result']}")
        else:
            print("🔸 No previous tool calls found.")

        print("\n🛒 User wants to buy an iPhone 15 Pro under 400 OMR")

        # Attempt 1: red iPhone
        args1 = {"text": "iphone 15 pro", "filters": {"color": "red", "price_range": {"max": 400}}}
        result1 = self.call_tool("search_products", args1)

        if result1['total_found'] == 0:
            print("❌ No products found in red. Retrying with black...")
            time.sleep(1)
            args2 = {"text": "iphone 15 pro", "filters": {"color": "black", "price_range": {"max": 400}}}
            result2 = self.call_tool("search_products", args2)

            if result2['total_found'] > 0:
                product = result2['products'][0]
                print(f"✅ Found product: {product['name']} for {product['price']} OMR")
                order_args = {"product_id": "abc123", "user_id": "U456"}
                result3 = self.call_tool("place_order", order_args)

                if result3['status'] == "success":
                    print(f"📦 Order placed! Order number: {result3['order_number']}")
                else:
                    print("❌ Failed to place order")
        else:
            print("✅ Found product in first try!")


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    memory = GraphMemory(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    session_id = "chat-001"

    agent = Agent(memory, session_id)
    agent.handle_user_request()

    memory.close()
