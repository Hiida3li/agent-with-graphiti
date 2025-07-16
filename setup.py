import os
import logging
import json
import uuid
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai
from abc import ABC, abstractmethod
import time
from quixstreams import Application
from string import Template

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def search_products(text: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Searches the product catalog for items based on a text query and optional filters.

    Args:
        text: A search query for products, like 'blue iphone' or 'samsung tv'.
        filters: Optional filters for color or price range to refine the search.
                 Expected format:
                 {
                    "attributes": {
                        "color": "red"
                    },
                    "price_range": {
                        "operation": "between", # "lt", "gt", "between"
                        "min": 100,
                        "max": 500
                    }
                 }

    Returns:
        A list of dictionaries, where each dictionary represents a product.
    """
    logger.info(f"Tool Call: search_products(text='{text}', filters={filters})")

    mock_products = [
        {
            "id": "iphone-15-pro-red",
            "name": "iPhone 15 Pro",
            "color": "Red",
            "price": 999.99,
            "category": "Electronics/Smartphones",
            "brand": "Apple",
            "in_stock": True,
            "description": "Latest iPhone 15 Pro in stunning red color"
        },
        {
            "id": "iphone-15-pro-blue",
            "name": "iPhone 15 Pro",
            "color": "Blue",
            "price": 999.99,
            "category": "Electronics/Smartphones",
            "brand": "Apple",
            "in_stock": True,
            "description": "Latest iPhone 15 Pro in deep blue color"
        },
        {
            "id": "samsung-galaxy-s24",
            "name": "Samsung Galaxy S24",
            "color": "Black",
            "price": 899.99,
            "category": "Electronics/Smartphones",
            "brand": "Samsung",
            "in_stock": False,
            "description": "Samsung Galaxy S24 in midnight black"
        }
    ]

    results = []
    if not text or text.strip() == "":
        results = mock_products.copy()
    else:
        search_terms = text.lower().split()
        for product in mock_products:
            product_text = f"{product['name']} {product['color']} {product['brand']} {product['description']}".lower()
            if any(term in product_text for term in search_terms):
                results.append(product)

    if filters and results:
        filtered_results = []
        for product in results:
            skip_product = False
            if filters.get("attributes", {}).get("color"):
                filter_color = filters["attributes"]["color"].lower()
                if filter_color not in product["color"].lower():
                    skip_product = True
            if not skip_product and filters.get("price_range"):
                price_filter = filters["price_range"]
                operation = price_filter.get("operation", "eq") # default to 'eq' for simplicity if not specified
                product_price = product["price"]

                if operation == "lt" and product_price >= price_filter.get("max", float('inf')):
                    skip_product = True
                elif operation == "gt" and product_price <= price_filter.get("min", 0):
                    skip_product = True
                elif operation == "between":
                    min_price = price_filter.get("min", 0)
                    max_price = price_filter.get("max", float('inf'))
                    if not (min_price <= product_price <= max_price):
                        skip_product = True
            if not skip_product:
                filtered_results.append(product)
        results = filtered_results

    return results

def search_faqs(text: str) -> List[Dict[str, str]]:
    """
    Searches the knowledge base for answers to frequently asked questions about topics
    like shipping, returns, warranty, and payment methods.

    Args:
        text: The user's question, e.g., 'how long does shipping take?' or
              'what is the return policy?'.

    Returns:
        A list of dictionaries, where each dictionary contains 'question' and 'answer' fields.
    """
    logger.info(f"Tool Call: search_faqs(text='{text}')")
    # Mock FAQ data
    mock_faqs = [
        {"question": "How long does shipping take?", "answer": "Standard shipping usually takes 3-5 business days."},
        {"question": "What is your return policy?", "answer": "You can return items within 30 days of purchase with a receipt."},
        {"question": "Do you offer warranty?", "answer": "Most electronics come with a 1-year manufacturer's warranty."},
        {"question": "What payment methods do you accept?", "answer": "We accept Visa, MasterCard, American Express, and PayPal."}
    ]

    results = []
    search_terms = text.lower().split()
    for faq in mock_faqs:
        faq_text = f"{faq['question']} {faq['answer']}".lower()
        if any(term in faq_text for term in search_terms):
            results.append(faq)
    return results


TOOLS = [search_products, search_faqs] # <--- Changed this line!

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # Pass the list of Python functions directly to the tools parameter
        self.model = genai.GenerativeModel(model_name, tools=TOOLS) # <--- No change needed here, it already uses TOOLS

    def generate(self, prompt: str) -> genai.types.GenerateContentResponse:
        logger.debug(f"Generating response for prompt length: {len(prompt)}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Gemini API call attempt {attempt + 1}")
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=10000,
                        top_p=0.95
                    )
                )

                if response and response.candidates:
                    logger.debug("Successfully received a valid response from Gemini.")
                    return response

                logger.warning(f"Empty response from Gemini on attempt {attempt + 1}.")

            except Exception as e:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                time.sleep(1)

        logger.error("All retry attempts to reach Gemini failed.")
        return None

def generate_final_response(context: Dict[str, Any], prompt_template: str) -> str:
    """Generate final LLM response with updated context including function execution results"""
    # Ensure this LLMProvider instance also has access to the tools for potential
    # multi-turn interaction or re-evaluation if needed during synthesis.
    # For now, it's just generating text, so tools aren't strictly necessary here.
    llm_provider = GeminiProvider()

    function_results = ""
    latest_interaction = context.get("history", [])[-1] if context.get("history") else None

    if latest_interaction:
        for execution in latest_interaction.get("function_executions", []):
            if execution.get("execution_status") == "completed":
                func_name = execution.get("function_name", "")
                result = execution.get("execution_result", {})

                if result is not None:
                    function_results += f"\nFunction: {func_name}\n"
                    if "formatted_summary" in result:
                        function_results += result["formatted_summary"]
                    else:
                        function_results += f"Result: {json.dumps(result, indent=2)}\n"

    template = Template(prompt_template)
    final_prompt = template.safe_substitute(
        query=context.get("query", ""),
        history=json.dumps(context.get("history", []), indent=2),
        function_results=function_results
    )

    logger.debug(f"Generating final response with prompt length: {len(final_prompt)}")
    final_response_object = llm_provider.generate(final_prompt)

    return final_response_object.text if final_response_object else "I'm sorry, I couldn't generate a final response."

def process_message(msg: Dict[str, Any], llm_provider: LLMProvider) -> List[Dict[str, Any]]:
    header = msg.get("header", {})
    payload = msg.get("payload", {})
    agent = payload.get("agent", {})
    context = agent.get("context", {})
    prompt_template = agent.get("prompt", "")

    template = Template(prompt_template)
    prompt = template.safe_substitute(**context)

    llm_response_object = llm_provider.generate(prompt)

    function_calls = []
    direct_response = ""
    llm_reasoning = ""

    if llm_response_object and llm_response_object.candidates:
        # Gemini's response can contain parts, including text or function_call
        for part in llm_response_object.candidates[0].content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args)
                })
                llm_reasoning = f"Decided to call function(s): {', '.join([fc['name'] for fc in function_calls])}"
            elif part.text:
                direct_response += part.text # Accumulate text parts
                llm_reasoning = "Provided a direct answer."
    else:
        direct_response = "I'm having trouble connecting. Please try again."
        llm_reasoning = "LLM response was empty or failed."

    if "history" not in context:
        context["history"] = []

    response_entry = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "user_query": context.get("query", ""),
        "llm_reasoning": llm_reasoning,
        "function_executions": []
    }

    messages = []

    if function_calls:
        first_func_call = function_calls[0]
        for i, func_call in enumerate(function_calls):
            execution_id = str(uuid.uuid4())
            function_execution = {
                "execution_id": execution_id,
                "function_name": func_call.get("name"),
                "parameters": func_call.get("args", {}),
                "execution_status": "pending" if i == 0 else "queued",
            }
            response_entry["function_executions"].append(function_execution)
            if i == 0:
                first_func_call["id"] = execution_id

        message = {
            "header": header,
            "payload": {
                "agent": {
                    "context": context,
                    "prompt": prompt_template,
                    "current_function_execution": first_func_call,
                    "remaining_function_calls": function_calls[1:] if len(function_calls) > 1 else []
                }
            }
        }
        messages.append(message)
        response_entry["response_type"] = "function_assisted"
        context["history"].append(response_entry)

    else:
        response_entry["response_type"] = "direct_knowledge"
        response_entry["direct_llm_response"] = direct_response
        response_entry["response"] = direct_response
        context["history"].append(response_entry)

        messages.append({
            "header": header,
            "payload": {
                "agent": {
                    "context": context,
                    "prompt": prompt_template,
                    "response_type": "direct_knowledge",
                    "response": direct_response
                }
            }
        })

    return messages

def process_function_response(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Process function call responses and update the structured hierarchy"""
    agent = msg.get("payload", {}).get("agent", {})
    function_call = agent.get("function_call", {})

    if "response" in function_call:
        func_call_id = function_call.get("id")
        function_name = function_call.get("name")
        logger.info(f"Function {function_name} (ID: {func_call_id}) completed successfully")

        context = agent.get("context", {})
        history = context.get("history", [])

        for interaction in history:
            for execution in interaction.get("function_executions", []):
                if execution.get("execution_id") == func_call_id:
                    execution["execution_status"] = "completed"
                    execution["completed_at"] = time.time()
                    execution["execution_result"] = function_call["response"]
                    break

        latest_interaction = history[-1] if history else None
        all_completed = False

        if latest_interaction:
            executions = latest_interaction.get("function_executions", [])
            all_completed = all(
                exec.get("execution_status") == "completed"
                for exec in executions
            )

            if all_completed:
                latest_interaction["execution_summary"] = {
                    "total_functions_executed": len(executions),
                    "all_successful": all(exec.get("error_details") is None for exec in executions),
                    "execution_results": [exec["execution_result"] for exec in executions if exec.get("execution_result")]
                }

                if latest_interaction.get("response_type") == "function_assisted":
                    logger.info("All function executions completed, generating final synthesized response")
                    final_llm_response = generate_final_response(context, agent.get("prompt", ""))
                    latest_interaction["response"] = final_llm_response
                else:
                    logger.info("Direct knowledge query completed, no synthesis needed")

        context["history"] = history

        final_msg = {
            "header": msg.get("header", {}),
            "payload": {
                "agent": {
                    "context": context,
                    "prompt": agent.get("prompt", ""),
                    "response": latest_interaction.get("response", "") if all_completed and latest_interaction else ""
                }
            }
        }
        return final_msg

    return msg

def main():
    app = Application(
        broker_address=os.getenv("KAFKA_BROKER", "localhost:9092"),
        consumer_group="llm-processor-group",
        auto_offset_reset="earliest"
    )

    llm_provider = GeminiProvider()

    request_topic = app.topic("agent-requests", value_deserializer="json")
    response_topic = app.topic("agent-function-responses", value_deserializer="json")

    producer = app.get_producer()

    def route_message(msg, target_topic):
        header = msg.get("header", {})
        partition_key = header.get("id", "")

        producer.produce(
            topic=target_topic,
            key=partition_key,
            value=json.dumps(msg)
        )
        logger.info(f"Message routed to topic: {target_topic}")

    def process_request(msg):
        messages = process_message(msg, llm_provider)

        for message in messages:
            function_call = message.get("payload", {}).get("agent", {}).get("current_function_execution")

            if function_call and "name" in function_call:
                route_message(message, function_call["name"])
            else:
                route_message(message, "agent-responses")

    def process_response(msg):
        result = process_function_response(msg)
        route_message(result, "agent-responses")

    request_stream = app.dataframe(request_topic)
    response_stream = app.dataframe(response_topic)

    request_stream.update(process_request)
    response_stream.update(process_response)

    logger.info("Starting LLM service...")
    app.run()

if __name__ == "__main__":
    main()
