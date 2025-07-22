# import os
# import logging
# import json
# import uuid
# from typing import Dict, Any, List
# from dotenv import load_dotenv
# import google.generativeai as genai
# from abc import ABC, abstractmethod
# import time
# from quixstreams import Application
# from string import Template
#
# load_dotenv()
#
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#
# if GOOGLE_API_KEY is None:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
#
# client = genai.Client(api_key=GOOGLE_API_KEY)
#
# search_products = {
#     "name": "search_products",
#     "description": "Searches the product catalog for items based on a text query and optional filters.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "query": {
#                 "type": "string",
#                 "description": "A search query for products."
#             },
#             "filters": {
#                 "type": "object",
#                 "description": "Optional filters like color.",
#                 "properties": {
#                     "color": {"type": "string"}
#                 }
#             }
#         },
#         "required": ["query"]
#     }
# }
#
# search_faqs = {
#     "name": "search_faqs",
#     "description": "Searches a knowledge base of Frequently Asked Questions (FAQs) based on a user's query. Returns a list of relevant FAQs including their questions, answers, and categories.",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "text": {
#                 "type": "string",
#                 "description": "The user's question or search query to find relevant FAQs. For example: 'How long does shipping take?' or 'return policy'."
#             }
#         },
#         "required": ["text"]
#     }
# }
# TOOLS = [search_products, search_faqs]
#
# class LLMProvider(ABC):
#     @abstractmethod
#     def generate(self, prompt: str) -> str:
#         pass
#
# class GeminiProvider(LLMProvider):
#     def __init__(self, model_name: str = "gemini-2.5-flash"):
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         self.model = genai.GenerativeModel(model_name, tools=TOOLS)
#
#     def generate(self, prompt: str) -> genai.types.GenerateContentResponse:
#         logger.debug(f"Generating response for prompt length: {len(prompt)}")
#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 logger.debug(f"Gemini API call attempt {attempt + 1}")
#                 response = self.model.generate_content(
#                     prompt,
#                     generation_config=genai.types.GenerationConfig(
#                         temperature=0,
#                         max_output_tokens=10000,
#                         top_p=0.95
#                     )
#                 )
#
#                 if response and response.candidates:
#                     logger.debug("Successfully received a valid response from Gemini.")
#                     return response
#
#                 logger.warning(f"Empty response from Gemini on attempt {attempt + 1}.")
#
#             except Exception as e:
#                 logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")
#
#             if attempt < max_retries - 1:
#                 time.sleep(1)
#
#         logger.error("All retry attempts to reach Gemini failed.")
#         return None
#
#
# def generate_final_response(context: Dict[str, Any], prompt_template: str) -> str:
#     """Generate final LLM response with updated context including function execution results"""
#     from string import Template
#
#     llm_provider = GeminiProvider()
#
#     function_results = ""
#     latest_interaction = context.get("interactions", [])[-1] if context.get("interactions") else None
#
#     if latest_interaction:
#         for execution in latest_interaction.get("function_executions", []):
#             if execution.get("execution_status") == "completed":
#                 func_name = execution.get("function_name", "")
#                 result = execution.get("execution_result", {})
#
#                 if result is not None:
#                     function_results += f"\nFunction: {func_name}\n"
#                     # Use the formatted_summary if the service provided it, otherwise use raw result
#                     if "formatted_summary" in result:
#                         function_results += result["formatted_summary"]
#                     else:
#                         function_results += f"Result: {json.dumps(result, indent=2)}\n"
#
#     template = Template(prompt_template)
#     final_prompt = template.safe_substitute(
#         query=context.get("query", ""),
#         interactions=json.dumps(context.get("interactions", []), indent=2),
#         history="\n".join(context.get("history", [])),
#         function_results=function_results
#     )
#
#     logger.debug(f"Generating final response with prompt length: {len(final_prompt)}")
#     final_response_object = llm_provider.generate(final_prompt)
#
#     return final_response_object.text if final_response_object else "I'm sorry, I couldn't generate a final response."
#
#
# def process_message(msg: Dict[str, Any], llm_provider: LLMProvider) -> List[Dict[str, Any]]:
#     header = msg.get("header", {})
#     payload = msg.get("payload", {})
#     agent = payload.get("agent", {})
#     context = agent.get("context", {})
#     prompt_template = agent.get("prompt", "")
#
#     template = Template(prompt_template)
#     prompt = template.safe_substitute(**context)
#
#     # --- Start of Corrected Section ---
#     llm_response_object = llm_provider.generate(prompt)
#
#     function_calls = []
#     direct_response = ""
#     llm_reasoning = ""
#
#     if llm_response_object and llm_response_object.candidates:
#         part = llm_response_object.candidates[0].content.parts[0]
#         if part.function_call:
#             function_calls.append({
#                 "name": part.function_call.name,
#                 "args": dict(part.function_call.args)
#             })
#             llm_reasoning = f"Decided to call function: {part.function_call.name}"
#         else:
#             direct_response = llm_response_object.text
#             llm_reasoning = "Provided a direct answer."
#     else:
#         direct_response = "I'm having trouble connecting. Please try again."
#         llm_reasoning = "LLM response was empty or failed."
#
#     # Initialize both interactions and history
#     if "interactions" not in context:
#         context["interactions"] = []
#     if "history" not in context:
#         context["history"] = []
#
#     response_entry = {
#         "interaction_id": str(uuid.uuid4()),
#         "timestamp": time.time(),
#         "user_query": context.get("query", ""),
#         "llm_reasoning": llm_reasoning,
#         "function_executions": []
#     }
#
#     messages = []
#
#     if function_calls:
#         first_func_call = function_calls[0]
#         for i, func_call in enumerate(function_calls):
#             execution_id = str(uuid.uuid4())
#             function_execution = {
#                 "execution_id": execution_id,
#                 "function_name": func_call.get("name"),
#                 "parameters": func_call.get("args", {}),
#                 "execution_status": "pending" if i == 0 else "queued",
#             }
#             response_entry["function_executions"].append(function_execution)
#             if i == 0:
#                 first_func_call["id"] = execution_id
#
#             # Add tool call to history log
#             args_str = ", ".join([f"{k}={v}" for k, v in func_call.get("args", {}).items()])
#             tool_call_log = f"Tool Call: {func_call.get('name')}({args_str})"
#             context["history"].append(tool_call_log)
#
#         message = {
#             "header": header,
#             "payload": {
#                 "agent": {
#                     "context": context,
#                     "prompt": prompt_template,
#                     "current_function_execution": first_func_call,
#                     "remaining_function_calls": function_calls[1:] if len(function_calls) > 1 else []
#                 }
#             }
#         }
#         messages.append(message)
#         response_entry["response_type"] = "function_assisted"
#         context["interactions"].append(response_entry)
#
#     else:
#         response_entry["response_type"] = "direct_knowledge"
#         response_entry["direct_llm_response"] = direct_response
#         response_entry["response"] = direct_response
#         context["interactions"].append(response_entry)
#
#         messages.append({
#             "header": header,
#             "payload": {
#                 "agent": {
#                     "context": context,
#                     "prompt": prompt_template,
#                     "response_type": "direct_knowledge",
#                     "response": direct_response
#                 }
#             }
#         })
#
#     return messages
#
#
# def process_function_response(msg: Dict[str, Any]) -> Dict[str, Any]:
#     """Process function call responses and update the structured hierarchy"""
#     import time
#
#     agent = msg.get("payload", {}).get("agent", {})
#     function_call = agent.get("function_call", {})
#
#     if "response" in function_call:
#         func_call_id = function_call.get("id")
#         function_name = function_call.get("name")
#         logger.info(f"Function {function_name} (ID: {func_call_id}) completed successfully")
#
#         # Get context and its interactions
#         context = agent.get("context", {})
#         interactions = context.get("interactions", [])
#
#         # Add tool response to history log
#         function_result = function_call["response"]
#         tool_response_log = f"Tool Response: {json.dumps(function_result)}"
#         context["history"].append(tool_response_log)
#
#         # Find and update the function execution in the context interactions
#         for interaction in interactions:
#             for execution in interaction.get("function_executions", []):
#                 if execution.get("execution_id") == func_call_id:
#                     execution["execution_status"] = "completed"
#                     execution["completed_at"] = time.time()
#                     execution["execution_result"] = function_result
#                     break
#
#         # Simply update the function execution status and context
#         latest_interaction = interactions[-1] if interactions else None
#         all_completed = False
#
#         if latest_interaction:
#             executions = latest_interaction.get("function_executions", [])
#             all_completed = all(
#                 exec.get("execution_status") == "completed"
#                 for exec in executions
#             )
#
#             if all_completed:
#                 # Create summary of all execution results
#                 latest_interaction["execution_summary"] = {
#                     "total_functions_executed": len(executions),
#                     "all_successful": all(exec.get("error_details") is None for exec in executions),
#                     "execution_results": [exec["execution_result"] for exec in executions if exec.get("execution_result")]
#                 }
#
#                 # Only generate final LLM response for function-assisted queries
#                 if latest_interaction.get("response_type") == "function_assisted":
#                     logger.info("All function executions completed, generating final synthesized response")
#                     # Use the same prompt template for synthesis
#                     final_llm_response = generate_final_response(context, agent.get("prompt", ""))
#                     latest_interaction["response"] = final_llm_response
#                 else:
#                     logger.info("Direct knowledge query completed, no synthesis needed")
#
#         # Update context with modified interactions
#         context["interactions"] = interactions
#
#         # Create final structured response
#         final_msg = {
#             "header": msg.get("header", {}),
#             "payload": {
#                 "agent": {
#                     "context": context,
#                     "prompt": agent.get("prompt", ""),
#                     "response": latest_interaction.get("response", "") if all_completed and latest_interaction else ""
#                 }
#             }
#         }
#         return final_msg
#
#     return msg
#
#
# def main():
#     app = Application(
#         broker_address=os.getenv("KAFKA_BROKER", "localhost:9092"),
#         consumer_group="llm-processor-group",
#         auto_offset_reset="earliest"
#     )
#
#     llm_provider = GeminiProvider()
#
#     request_topic = app.topic("agent-requests", value_deserializer="json")
#     response_topic = app.topic("agent-function-responses", value_deserializer="json")
#
#     producer = app.get_producer()
#
#     def route_message(msg, target_topic):
#         """Route message to specific topic"""
#         header = msg.get("header", {})
#         partition_key = header.get("id", "")
#
#         producer.produce(
#             topic=target_topic,
#             key=partition_key,
#             value=json.dumps(msg)
#         )
#         logger.info(f"Message routed to topic: {target_topic}")
#
#     def process_request(msg):
#         """Process initial agent requests"""
#         messages = process_message(msg, llm_provider)
#
#         for message in messages:
#             function_call = message.get("payload", {}).get("agent", {}).get("current_function_execution")
#
#             if function_call and "name" in function_call:
#                 route_message(message, function_call["name"])
#             else:
#                 route_message(message, "agent-responses")
#
#     def process_response(msg):
#         """Process function call responses"""
#         result = process_function_response(msg)
#
#         route_message(result, "agent-responses")
#
#     request_stream = app.dataframe(request_topic)
#     response_stream = app.dataframe(response_topic)
#
#     request_stream.update(process_request)
#     response_stream.update(process_response)
#
#     logger.info("Starting LLM service...")
#     app.run()
#
# if __name__ == "__main__":
#     main()
#
#





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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")


search_products = {
    "name": "search_products",
    "description": "Searches the product catalog for items based on a text query and optional filters.",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "query": {
                "type_": "STRING",
                "description": "A search query for products."
            },
            "filters": {
                "type_": "OBJECT",
                "description": "Optional filters like color.",
                "properties": {
                    "color": {
                        "type_": "STRING"
                    }
                }
            }
        },
        "required": ["query"]
    }
}

search_faqs = {
    "name": "search_faqs",
    "description": "Searches a knowledge base of Frequently Asked Questions (FAQs) based on a user's query. Returns a list of relevant FAQs including their questions, answers, and categories.",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "text": {
                "type_": "STRING",
                "description": "The user's question or search query to find relevant FAQs. For example: 'How long does shipping take?' or 'return policy'."
            }
        },
        "required": ["text"]
    }
}

final_response = {
    "name": "final_response",
    "description": "Processes a final response request and passes the content through to the user. This tool handles the final formatting and routing of the message, whether it's a direct answer or derived from a previous function call.",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "content": {
                "type_": "STRING",
                "description": "The final content of the response to be formatted and sent to the user."
            }
        },
        "required": ["content"]
    }
}


TOOLS = [search_products, search_faqs, final_response]

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name, tools=TOOLS)

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
    from string import Template

    llm_provider = GeminiProvider()

    function_results = ""
    latest_interaction = context.get("interactions", [])[-1] if context.get("interactions") else None

    if latest_interaction:
        for execution in latest_interaction.get("function_executions", []):
            if execution.get("execution_status") == "completed":
                func_name = execution.get("function_name", "")
                result = execution.get("execution_result", {})

                if result is not None:
                    function_results += f"\nFunction: {func_name}\n"
                    # Use the formatted_summary if the service provided it, otherwise use raw result
                    if "formatted_summary" in result:
                        function_results += result["formatted_summary"]
                    else:
                        function_results += f"Result: {json.dumps(result, indent=2)}\n"

    template = Template(prompt_template)
    final_prompt = template.safe_substitute(
        query=context.get("query", ""),
        interactions=json.dumps(context.get("interactions", []), indent=2),
        history="\n".join(context.get("history", [])),
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

    # --- Start of Corrected Section ---
    llm_response_object = llm_provider.generate(prompt)

    function_calls = []
    direct_response = ""
    llm_reasoning = ""

    if llm_response_object and llm_response_object.candidates:
        part = llm_response_object.candidates[0].content.parts[0]
        if part.function_call:
            function_calls.append({
                "name": part.function_call.name,
                "args": dict(part.function_call.args)
            })
            llm_reasoning = f"Decided to call function: {part.function_call.name}"
        else:
            direct_response = llm_response_object.text
            llm_reasoning = "Provided a direct answer."
    else:
        direct_response = "I'm having trouble connecting. Please try again."
        llm_reasoning = "LLM response was empty or failed."

    # Initialize both interactions and history
    if "interactions" not in context:
        context["interactions"] = []
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

            # Add tool call to history log
            args_str = ", ".join([f"{k}={v}" for k, v in func_call.get("args", {}).items()])
            tool_call_log = f"Tool Call: {func_call.get('name')}({args_str})"
            context["history"].append(tool_call_log)

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
        context["interactions"].append(response_entry)

    else:
        response_entry["response_type"] = "direct_knowledge"
        response_entry["direct_llm_response"] = direct_response
        response_entry["response"] = direct_response
        context["interactions"].append(response_entry)

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
    import time

    agent = msg.get("payload", {}).get("agent", {})
    function_call = agent.get("function_call", {})

    if "response" in function_call:
        func_call_id = function_call.get("id")
        function_name = function_call.get("name")
        logger.info(f"Function {function_name} (ID: {func_call_id}) completed successfully")

        # Get context and its interactions
        context = agent.get("context", {})
        interactions = context.get("interactions", [])

        # Add tool response to history log
        function_result = function_call["response"]
        tool_response_log = f"Tool Response: {json.dumps(function_result)}"
        context["history"].append(tool_response_log)

        # Find and update the function execution in the context interactions
        for interaction in interactions:
            for execution in interaction.get("function_executions", []):
                if execution.get("execution_id") == func_call_id:
                    execution["execution_status"] = "completed"
                    execution["completed_at"] = time.time()
                    execution["execution_result"] = function_result
                    break

        # Simply update the function execution status and context
        latest_interaction = interactions[-1] if interactions else None
        all_completed = False

        if latest_interaction:
            executions = latest_interaction.get("function_executions", [])
            all_completed = all(
                exec.get("execution_status") == "completed"
                for exec in executions
            )

            if all_completed:
                # Create summary of all execution results
                latest_interaction["execution_summary"] = {
                    "total_functions_executed": len(executions),
                    "all_successful": all(exec.get("error_details") is None for exec in executions),
                    "execution_results": [exec["execution_result"] for exec in executions if exec.get("execution_result")]
                }

                # Only generate final LLM response for function-assisted queries
                if latest_interaction.get("response_type") == "function_assisted":
                    logger.info("All function executions completed, generating final synthesized response")
                    # Use the same prompt template for synthesis
                    final_llm_response = generate_final_response(context, agent.get("prompt", ""))
                    latest_interaction["response"] = final_llm_response
                else:
                    logger.info("Direct knowledge query completed, no synthesis needed")

        # Update context with modified interactions
        context["interactions"] = interactions

        # Create final structured response
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
        """Route message to specific topic"""
        header = msg.get("header", {})
        partition_key = header.get("id", "")

        producer.produce(
            topic=target_topic,
            key=partition_key,
            value=json.dumps(msg)
        )
        logger.info(f"Message routed to topic: {target_topic}")

    def process_request(msg):
        """Process initial agent requests"""
        messages = process_message(msg, llm_provider)

        for message in messages:
            function_call = message.get("payload", {}).get("agent", {}).get("current_function_execution")

            if function_call and "name" in function_call:
                route_message(message, function_call["name"])
            else:
                route_message(message, "agent-responses")

    def process_response(msg):
        """Process function call responses"""
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

