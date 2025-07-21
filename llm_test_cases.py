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
# # client = genai.Client(api_key=GOOGLE_API_KEY)  # Remove this line - not needed
#
# search_products = {
#     "name": "search_products",
#     "description": "Searches the product catalog for items based on a text query and optional filters.",
#     "parameters": {
#         "type_": "OBJECT",  # Required for parameters object
#         "properties": {
#             "query": {
#                 "type_": "STRING",
#                 "description": "A search query for products."
#             },
#             "filters": {
#                 "type_": "OBJECT",
#                 "description": "Optional filters like color.",
#                 "properties": {
#                     "color": {"type_": "STRING"}
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
#         "type_": "OBJECT",
#         "properties": {
#             "text": {
#                 "type_": "STRING",
#                 "description": "The user's question or search query to find relevant FAQs. For example: 'How long does shipping take?' or 'return policy'."
#             }
#         },
#         "required": ["text"]
#     }
# }
# TOOLS = [search_products, search_faqs]
#
#
# class LLMProvider(ABC):
#     @abstractmethod
#     def generate(self, prompt: str) -> str:
#         pass
#
#
# class GeminiProvider(LLMProvider):
#     def __init__(self, model_name: str = "gemini-2.5-flash"):
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#
#         self.model = genai.GenerativeModel(model_name, tools=TOOLS)
#         logger.info(f"GeminiProvider initialized with model: {model_name}")
#         logger.info(f"Tools re-enabled: {[tool['name'] for tool in TOOLS]}")
#
#     def generate(self, prompt: str) -> genai.types.GenerateContentResponse:
#         logger.debug(f"=== GEMINI REQUEST ===")
#         logger.debug(f"Prompt length: {len(prompt)}")
#         logger.debug(f"Prompt content: {prompt[:500]}...")  # Log first 500 chars
#
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
#                 logger.debug(f"=== GEMINI RESPONSE DEBUG ===")
#                 logger.debug(f"Response object exists: {response is not None}")
#
#                 if response:
#                     logger.debug(
#                         f"Has candidates: {hasattr(response, 'candidates') and response.candidates is not None}")
#                     logger.debug(f"Candidates count: {len(response.candidates) if response.candidates else 0}")
#
#                     if response.candidates:
#                         candidate = response.candidates[0]
#                         logger.debug(
#                             f"First candidate content: {hasattr(candidate, 'content') and candidate.content is not None}")
#
#                         if candidate.content and candidate.content.parts:
#                             logger.debug(f"Parts count: {len(candidate.content.parts)}")
#                             part = candidate.content.parts[0]
#
#                             # Check what's in the part
#                             logger.debug(
#                                 f"Part has function_call: {hasattr(part, 'function_call') and part.function_call is not None}")
#                             logger.debug(f"Part has text: {hasattr(part, 'text') and part.text is not None}")
#
#                             if hasattr(part, 'function_call') and part.function_call:
#                                 logger.debug(f"Function call name: {part.function_call.name}")
#                                 logger.debug(f"Function call args: {dict(part.function_call.args)}")
#
#                             if hasattr(part, 'text') and part.text:
#                                 logger.debug(f"Text content: {part.text[:200]}...")
#
#                     # Also check the response.text attribute
#                     try:
#                         response_text = response.text
#                         logger.debug(f"Response.text exists: {response_text is not None}")
#                         if response_text:
#                             logger.debug(f"Response.text content: {response_text[:200]}...")
#                     except Exception as text_error:
#                         logger.debug(f"Error accessing response.text: {text_error}")
#
#                 if response and response.candidates:
#                     logger.debug("Successfully received a valid response from Gemini.")
#                     return response
#
#                 logger.warning(f"Empty response from Gemini on attempt {attempt + 1}.")
#
#             except Exception as e:
#                 logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")
#                 logger.error(f"Exception type: {type(e)}")
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
#     llm_provider = GeminiProvider("gemini-2.5-flash")
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
#     logger.debug(f"=== PROCESSING MESSAGE ===")
#     logger.debug(f"User query: {context.get('query', 'NO QUERY')}")
#
#     template = Template(prompt_template)
#     # Add the user query explicitly to the prompt
#     enhanced_context = context.copy()
#     enhanced_context['user_query'] = context.get('query', '')
#
#     prompt = template.safe_substitute(**enhanced_context)
#
#     # If prompt doesn't include user query, add it explicitly
#     user_query = context.get('query', '')
#     if user_query and user_query not in prompt:
#         prompt += f"\n\nUser Query: {user_query}\n\nIMPORTANT: If the user is asking about products (like iPhone, phone, laptop, etc.), you MUST use the search_products function. If they're asking about shipping, returns, policies, etc., use the search_faqs function. Always use the appropriate function call rather than guessing or making assumptions."
#
#     logger.debug(f"=== FINAL PROMPT TO LLM ===")
#     logger.debug(f"Prompt: {prompt}")
#
#     # --- Start of Corrected Section ---
#     llm_response_object = llm_provider.generate(prompt)
#
#     function_calls = []
#     direct_response = ""
#     llm_reasoning = ""
#
#     logger.debug(f"=== RESPONSE PROCESSING ===")
#     logger.debug(f"LLM response object exists: {llm_response_object is not None}")
#
#     if llm_response_object and llm_response_object.candidates:
#         logger.debug("Response has candidates, processing...")
#         candidate = llm_response_object.candidates[0]
#
#         if candidate.content and candidate.content.parts:
#             part = candidate.content.parts[0]
#             logger.debug(
#                 f"Processing part with function_call: {hasattr(part, 'function_call') and part.function_call is not None}")
#
#             # Enhanced function call debugging
#             if hasattr(part, 'function_call'):
#                 logger.debug(f"function_call exists: {part.function_call is not None}")
#                 if part.function_call:
#                     logger.debug(f"function_call object: {part.function_call}")
#                     logger.debug(f"function_call type: {type(part.function_call)}")
#                     logger.debug(f"function_call dir: {dir(part.function_call)}")
#                     logger.debug(f"function_call has name attr: {hasattr(part.function_call, 'name')}")
#                     if hasattr(part.function_call, 'name'):
#                         logger.debug(f"function_call.name: '{part.function_call.name}'")
#                         logger.debug(f"function_call.name type: {type(part.function_call.name)}")
#                         logger.debug(f"function_call.name is truthy: {bool(part.function_call.name)}")
#                     logger.debug(f"function_call has args attr: {hasattr(part.function_call, 'args')}")
#                     if hasattr(part.function_call, 'args'):
#                         logger.debug(f"function_call.args: {part.function_call.args}")
#                         logger.debug(f"function_call.args type: {type(part.function_call.args)}")
#
#             if hasattr(part, 'function_call') and part.function_call and hasattr(part.function_call,
#                                                                                  'name') and part.function_call.name:
#                 logger.debug("=== FUNCTION CALL DETECTED ===")
#                 func_call = {
#                     "name": part.function_call.name,
#                     "args": dict(part.function_call.args) if part.function_call.args else {}
#                 }
#                 function_calls.append(func_call)
#                 llm_reasoning = f"Decided to call function: {part.function_call.name}"
#                 logger.debug(f"Function call: {func_call}")
#             else:
#                 logger.debug("=== NO FUNCTION CALL, CHECKING TEXT ===")
#                 try:
#                     # Try multiple ways to get text
#                     text_content = None
#
#                     # Method 1: response.text
#                     try:
#                         text_content = llm_response_object.text
#                         logger.debug(f"Got text via response.text: {text_content is not None}")
#                     except Exception as e:
#                         logger.debug(f"response.text failed: {e}")
#
#                     # Method 2: part.text
#                     if not text_content and hasattr(part, 'text'):
#                         text_content = part.text
#                         logger.debug(f"Got text via part.text: {text_content is not None}")
#
#                     if text_content:
#                         direct_response = text_content
#                         llm_reasoning = "Provided a direct answer."
#                         logger.debug(f"Direct response: {direct_response[:200]}...")
#                     else:
#                         logger.warning("No text content found in response")
#                         direct_response = "I'm experiencing technical issues. Please try again."
#                         llm_reasoning = "No text content in LLM response."
#
#                 except Exception as e:
#                     logger.error(f"Error extracting text from response: {e}")
#                     direct_response = "I'm experiencing technical issues. Please try again."
#                     llm_reasoning = f"Error extracting text: {e}"
#         else:
#             logger.warning("Response candidate has no content or parts")
#             direct_response = "I'm experiencing technical issues. Please try again."
#             llm_reasoning = "Response candidate has no content or parts."
#     else:
#         logger.error("No valid response from Gemini")
#         direct_response = "I'm having trouble connecting. Please try again."
#         llm_reasoning = "LLM response was empty or failed."
#
#     logger.debug(f"=== FINAL PROCESSING RESULT ===")
#     logger.debug(f"Function calls count: {len(function_calls)}")
#     logger.debug(f"Direct response: {direct_response[:100]}...")
#     logger.debug(f"LLM reasoning: {llm_reasoning}")
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
#         logger.info(f"Processing {len(function_calls)} function calls")
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
#         logger.info(f"No function calls, providing direct response")
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
#                     "execution_results": [exec["execution_result"] for exec in executions if
#                                           exec.get("execution_result")]
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
#         consumer_group=f"llm-processor-group-{int(time.time())}",  # Fresh consumer group
#         auto_offset_reset="earliest"
#     )
#
#     llm_provider = GeminiProvider("gemini-1.5-flash")
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
#         logger.info(f" RECEIVED MESSAGE ON agent-requests: {json.dumps(msg, indent=2)}")
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
#
# if __name__ == "__main__":
#     main()




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
                    "color": {"type_": "STRING"}
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

format_response = {
    "name": "format_response",
    "description": "Formats the final response to the customer in a professional and consistent manner. This should be called for every customer response.",
    "parameters": {
        "type_": "OBJECT",
        "properties": {
            "content": {
                "type_": "STRING",
                "description": "The main content of the response to be formatted for the customer."
            },
            "response_type": {
                "type_": "STRING",
                "description": "The type of response: 'standard', 'greeting', 'closing', or 'error'."
            }
        },
        "required": ["content"]
    }
}

TOOLS = [search_products, search_faqs, format_response]


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        self.model = genai.GenerativeModel(model_name, tools=TOOLS)
        logger.info(f"GeminiProvider initialized with model: {model_name}")
        logger.info(f"Tools available: {[tool['name'] for tool in TOOLS]}")

    def generate(self, prompt: str) -> genai.types.GenerateContentResponse:
        logger.debug(f"=== GEMINI REQUEST ===")
        logger.debug(f"Prompt length: {len(prompt)}")
        logger.debug(f"Prompt content: {prompt[:500]}...")

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

                logger.debug(f"=== GEMINI RESPONSE DEBUG ===")
                logger.debug(f"Response object exists: {response is not None}")

                if response:
                    logger.debug(
                        f"Has candidates: {hasattr(response, 'candidates') and response.candidates is not None}")
                    logger.debug(f"Candidates count: {len(response.candidates) if response.candidates else 0}")

                    if response.candidates:
                        candidate = response.candidates[0]
                        logger.debug(
                            f"First candidate content: {hasattr(candidate, 'content') and candidate.content is not None}")

                        if candidate.content and candidate.content.parts:
                            logger.debug(f"Parts count: {len(candidate.content.parts)}")
                            part = candidate.content.parts[0]

                            logger.debug(
                                f"Part has function_call: {hasattr(part, 'function_call') and part.function_call is not None}")
                            logger.debug(f"Part has text: {hasattr(part, 'text') and part.text is not None}")

                            if hasattr(part, 'function_call') and part.function_call:
                                logger.debug(f"Function call name: {part.function_call.name}")
                                logger.debug(f"Function call args: {dict(part.function_call.args)}")

                            if hasattr(part, 'text') and part.text:
                                logger.debug(f"Text content: {part.text[:200]}...")

                    try:
                        response_text = response.text
                        logger.debug(f"Response.text exists: {response_text is not None}")
                        if response_text:
                            logger.debug(f"Response.text content: {response_text[:200]}...")
                    except Exception as text_error:
                        logger.debug(f"Error accessing response.text: {text_error}")

                if response and response.candidates:
                    logger.debug("Successfully received a valid response from Gemini.")
                    return response

                logger.warning(f"Empty response from Gemini on attempt {attempt + 1}.")

            except Exception as e:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")
                logger.error(f"Exception type: {type(e)}")

            if attempt < max_retries - 1:
                time.sleep(1)

        logger.error("All retry attempts to reach Gemini failed.")
        return None


def generate_final_response(context: Dict[str, Any], prompt_template: str) -> str:
    """Generate final LLM response with updated context including function execution results"""
    from string import Template

    llm_provider = GeminiProvider("gemini-2.5-flash")

    function_results = ""
    latest_interaction = context.get("interactions", [])[-1] if context.get("interactions") else None

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
        interactions=json.dumps(context.get("interactions", []), indent=2),
        history="\n".join(context.get("history", [])),
        function_results=function_results
    )

    # Add instruction to format the response
    final_prompt += "\n\nIMPORTANT: After providing your response, you MUST call the format_response function to format it properly for the customer."

    logger.debug(f"Generating final response with prompt length: {len(final_prompt)}")
    final_response_object = llm_provider.generate(final_prompt)

    return final_response_object.text if final_response_object else "I'm sorry, I couldn't generate a final response."


def process_message(msg: Dict[str, Any], llm_provider: LLMProvider) -> List[Dict[str, Any]]:
    header = msg.get("header", {})
    payload = msg.get("payload", {})
    agent = payload.get("agent", {})
    context = agent.get("context", {})
    prompt_template = agent.get("prompt", "")

    logger.debug(f"=== PROCESSING MESSAGE ===")
    logger.debug(f"User query: {context.get('query', 'NO QUERY')}")

    template = Template(prompt_template)
    enhanced_context = context.copy()
    enhanced_context['user_query'] = context.get('query', '')

    prompt = template.safe_substitute(**enhanced_context)

    user_query = context.get('query', '')
    if user_query and user_query not in prompt:
        prompt += f"\n\nUser Query: {user_query}\n\nIMPORTANT: If the user is asking about products (like iPhone, phone, laptop, etc.), you MUST use the search_products function. If they're asking about shipping, returns, policies, etc., use the search_faqs function. Always use the appropriate function call rather than guessing or making assumptions. After providing any response, you MUST call format_response to format it for the customer."

    logger.debug(f"=== FINAL PROMPT TO LLM ===")
    logger.debug(f"Prompt: {prompt}")

    llm_response_object = llm_provider.generate(prompt)

    function_calls = []
    direct_response = ""
    llm_reasoning = ""

    logger.debug(f"=== RESPONSE PROCESSING ===")
    logger.debug(f"LLM response object exists: {llm_response_object is not None}")

    if llm_response_object and llm_response_object.candidates:
        logger.debug("Response has candidates, processing...")
        candidate = llm_response_object.candidates[0]

        if candidate.content and candidate.content.parts:
            part = candidate.content.parts[0]
            logger.debug(
                f"Processing part with function_call: {hasattr(part, 'function_call') and part.function_call is not None}")

            if hasattr(part, 'function_call'):
                logger.debug(f"function_call exists: {part.function_call is not None}")
                if part.function_call:
                    logger.debug(f"function_call object: {part.function_call}")
                    logger.debug(f"function_call has name attr: {hasattr(part.function_call, 'name')}")
                    if hasattr(part.function_call, 'name'):
                        logger.debug(f"function_call.name: '{part.function_call.name}'")
                    logger.debug(f"function_call has args attr: {hasattr(part.function_call, 'args')}")
                    if hasattr(part.function_call, 'args'):
                        logger.debug(f"function_call.args: {part.function_call.args}")

            if hasattr(part, 'function_call') and part.function_call and hasattr(part.function_call,
                                                                                 'name') and part.function_call.name:
                logger.debug("=== FUNCTION CALL DETECTED ===")
                func_call = {
                    "name": part.function_call.name,
                    "args": dict(part.function_call.args) if part.function_call.args else {}
                }
                function_calls.append(func_call)
                llm_reasoning = f"Decided to call function: {part.function_call.name}"
                logger.debug(f"Function call: {func_call}")
            else:
                logger.debug("=== NO FUNCTION CALL, CHECKING TEXT ===")
                try:
                    text_content = None

                    try:
                        text_content = llm_response_object.text
                        logger.debug(f"Got text via response.text: {text_content is not None}")
                    except Exception as e:
                        logger.debug(f"response.text failed: {e}")

                    if not text_content and hasattr(part, 'text'):
                        text_content = part.text
                        logger.debug(f"Got text via part.text: {text_content is not None}")

                    if text_content:
                        # Create format_response function call for direct responses
                        format_func_call = {
                            "name": "format_response",
                            "args": {
                                "content": text_content,
                                "response_type": "standard"
                            }
                        }
                        function_calls.append(format_func_call)
                        llm_reasoning = "Provided a direct answer, formatting it for customer."
                        logger.debug(f"Auto-created format function call for direct response")
                    else:
                        logger.warning("No text content found in response")
                        format_func_call = {
                            "name": "format_response",
                            "args": {
                                "content": "I'm experiencing technical issues. Please try again.",
                                "response_type": "error"
                            }
                        }
                        function_calls.append(format_func_call)
                        llm_reasoning = "No text content in LLM response, using error message."

                except Exception as e:
                    logger.error(f"Error extracting text from response: {e}")
                    format_func_call = {
                        "name": "format_response",
                        "args": {
                            "content": "I'm experiencing technical issues. Please try again.",
                            "response_type": "error"
                        }
                    }
                    function_calls.append(format_func_call)
                    llm_reasoning = f"Error extracting text: {e}"
        else:
            logger.warning("Response candidate has no content or parts")
            format_func_call = {
                "name": "format_response",
                "args": {
                    "content": "I'm experiencing technical issues. Please try again.",
                    "response_type": "error"
                }
            }
            function_calls.append(format_func_call)
            llm_reasoning = "Response candidate has no content or parts."
    else:
        logger.error("No valid response from Gemini")
        format_func_call = {
            "name": "format_response",
            "args": {
                "content": "I'm having trouble connecting. Please try again.",
                "response_type": "error"
            }
        }
        function_calls.append(format_func_call)
        llm_reasoning = "LLM response was empty or failed."

    logger.debug(f"=== FINAL PROCESSING RESULT ===")
    logger.debug(f"Function calls count: {len(function_calls)}")
    logger.debug(f"LLM reasoning: {llm_reasoning}")

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
        logger.info(f"Processing {len(function_calls)} function calls")
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

        context = agent.get("context", {})
        interactions = context.get("interactions", [])

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

        latest_interaction = interactions[-1] if interactions else None
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
                    "execution_results": [exec["execution_result"] for exec in executions if
                                          exec.get("execution_result")]
                }

                # Check if the last execution was format_response
                last_execution = executions[-1] if executions else None
                if last_execution and last_execution.get("function_name") == "format_response":
                    # Use the formatted content as the final response
                    format_result = last_execution.get("execution_result", {})
                    latest_interaction["response"] = format_result.get("formatted_content", "")
                    logger.info("Used formatted response as final response")
                else:
                    # Need to call format_response for the final response
                    logger.info("Generating final synthesized response and formatting it")
                    final_llm_response = generate_final_response(context, agent.get("prompt", ""))

                    # Create format_response function call
                    format_func_call = {
                        "name": "format_response",
                        "args": {
                            "content": final_llm_response,
                            "response_type": "standard"
                        },
                        "id": str(uuid.uuid4())
                    }

                    # Add this as a new function execution
                    new_execution = {
                        "execution_id": format_func_call["id"],
                        "function_name": format_func_call["name"],
                        "parameters": format_func_call["args"],
                        "execution_status": "pending"
                    }
                    latest_interaction["function_executions"].append(new_execution)

                    # Create message to call format_response
                    format_message = {
                        "header": msg.get("header", {}),
                        "payload": {
                            "agent": {
                                "context": context,
                                "prompt": agent.get("prompt", ""),
                                "current_function_execution": format_func_call
                            }
                        }
                    }
                    return format_message

        context["interactions"] = interactions

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
        consumer_group=f"llm-processor-group-{int(time.time())}",
        auto_offset_reset="earliest"
    )

    llm_provider = GeminiProvider("gemini-1.5-flash")

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
        logger.info(f" RECEIVED MESSAGE ON agent-requests: {json.dumps(msg, indent=2)}")
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

        # Check if result is a new function call message (for format_response)
        function_call = result.get("payload", {}).get("agent", {}).get("current_function_execution")
        if function_call and "name" in function_call:
            route_message(result, function_call["name"])
        else:
            route_message(result, "agent-responses")

    request_stream = app.dataframe(request_topic)
    response_stream = app.dataframe(response_topic)

    request_stream.update(process_request)
    response_stream.update(process_response)

    logger.info("Starting LLM service...")
    app.run()


if __name__ == "__main__":
    main()
