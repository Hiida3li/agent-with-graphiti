import os
import logging
import json
import uuid
from typing import Dict, Any, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
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

search_products_function = types.FunctionDeclaration(
    name="search_products",
    description="Searches the product catalog for items based on a text query and optional filters.",
    parameters_json_schema={
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
                    "color": {
                        "type": "string"
                    }
                }
            }
        },
        "required": ["query"]
    }
)

search_faqs_function = types.FunctionDeclaration(
    name="search_faqs",
    description="Searches a knowledge base of Frequently Asked Questions (FAQs) based on a user's query. Returns a list of relevant FAQs including their questions, answers, and categories.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The user's question or search query to find relevant FAQs. For example: 'How long does shipping take?' or 'return policy'."
            }
        },
        "required": ["text"]
    }
)

respond_to_user_function = types.FunctionDeclaration(
    name="respond_to_user",
    description="Use this tool to send a response directly to the user. This tool also handles the formatting and routing of the message, and allows the AI to incorporate all available information to send an informative response that includes thoughts, function calls and instructions.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The response to be sent to the user"
            }
        },
        "required": ["content"]
    }
)

TOOLS = [
    types.Tool(function_declarations=[
        search_products_function,
        search_faqs_function,
        respond_to_user_function
    ])
]


def extract_function_args(function_call_args):
    """Extract function call arguments in JSON-serializable format for new SDK"""
    try:

        if isinstance(function_call_args, dict):

            logger.debug("Function args in direct dictionary format")
            return function_call_args


        if hasattr(function_call_args, 'dict'):
            logger.debug("Function args in Pydantic model format")
            return function_call_args.dict()

        # Handle if it's an object with direct attribute access
        if hasattr(function_call_args, '__dict__'):
            logger.debug("Function args in object format")
            return function_call_args.__dict__

        # Legacy format handling (keep for backward compatibility)
        if hasattr(function_call_args, 'items'):
            logger.debug("Function args in legacy format - converting")
            args_dict = {}
            for key, value in function_call_args.items():
                if hasattr(value, 'string_value'):
                    args_dict[key] = value.string_value
                elif hasattr(value, 'struct_value'):
                    # Handle nested structures like filters
                    nested_dict = {}
                    for nested_key, nested_value in value.struct_value.fields.items():
                        if hasattr(nested_value, 'string_value'):
                            nested_dict[nested_key] = nested_value.string_value
                        else:
                            nested_dict[nested_key] = str(nested_value)
                    args_dict[key] = nested_dict
                elif hasattr(value, 'number_value'):
                    args_dict[key] = value.number_value
                elif hasattr(value, 'bool_value'):
                    args_dict[key] = value.bool_value
                else:
                    # Fallback to string conversion
                    args_dict[key] = str(value)
            return args_dict

        # Final fallback - try to convert to string and parse as JSON
        logger.warning(f"Unexpected function args format: {type(function_call_args)}")
        return json.loads(str(function_call_args)) if str(function_call_args) else {}

    except Exception as e:
        logger.warning(f"Could not extract function args: {e}")
        logger.warning(f"Args type: {type(function_call_args)}")
        logger.warning(f"Args content: {function_call_args}")
        return {}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str):
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        try:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")


            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name

            logger.debug(f"GeminiProvider initialized successfully:")
            logger.debug(f"  - Model: {model_name}")
            logger.debug(f"  - Client type: {type(self.client)}")
            logger.debug(
                f"  - Tools configured: {len(TOOLS)} tool groups with functions: {[func.name for tool in TOOLS for func in tool.function_declarations]}")

        except Exception as e:
            logger.error(f"Failed to initialize GeminiProvider: {e}")
            logger.error(f"Available environment variables:")
            logger.error(f"  - GOOGLE_API_KEY: {'***SET***' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
            logger.error(f"  - GEMINI_API_KEY: {'***SET***' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
            raise

    def generate(self, prompt: str):
        logger.debug(f"Generating response for prompt length: {len(prompt)}")
        logger.debug(f"Prompt preview (first 200 chars): {prompt[:200]}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Gemini API call attempt {attempt + 1}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=TOOLS,
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(mode='ANY')
                        ),
                        temperature=0,
                        max_output_tokens=10000,
                        top_p=0.95
                    )
                )

                if response and response.candidates:
                    logger.debug("Successfully received a valid response from Gemini.")
                    logger.debug(f"Response has {len(response.candidates)} candidates")
                    return response

                logger.warning(f"Empty response from Gemini on attempt {attempt + 1}.")

            except errors.APIError as e:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e.code} - {e.message}")
            except Exception as e:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                time.sleep(1)

        logger.error("All retry attempts to reach Gemini failed.")
        return None


def generate_final_response(context: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    """Generate final LLM response - ONLY returns function call info (Mode 2 only)"""
    logger.debug("Starting final response generation")
    from string import Template

    llm_provider = GeminiProvider()

    function_results = ""
    latest_interaction = context.get("interactions", [])[-1] if context.get("interactions") else None

    logger.debug(f"Context has {len(context.get('interactions', []))} interactions")
    logger.debug(f"Latest interaction: {latest_interaction.get('interaction_id') if latest_interaction else 'None'}")

    if latest_interaction:
        logger.debug(f"Processing {len(latest_interaction.get('function_executions', []))} function executions")
        for execution in latest_interaction.get("function_executions", []):
            if execution.get("execution_status") == "completed":
                func_name = execution.get("function_name", "")
                result = execution.get("execution_result", {})
                logger.debug(f"Including result from function: {func_name}")

                if result is not None:
                    function_results += f"\nFunction: {func_name}\n"
                    # Use the formatted_summary if the service provided it, otherwise use raw result
                    if "formatted_summary" in result:
                        function_results += result["formatted_summary"]
                    else:
                        function_results += f"Result: {json.dumps(result, indent=2)}\n"

    logger.debug(f"Function results length: {len(function_results)} chars")

    template = Template(prompt_template)
    final_prompt = template.safe_substitute(
        query=context.get("query", ""),
        interactions=json.dumps(context.get("interactions", []), indent=2),
        history="\n".join(context.get("history", [])),
        function_results=function_results
    )

    logger.debug(f"Generating final response with prompt length: {len(final_prompt)}")
    final_response_object = llm_provider.generate(final_prompt)

    if final_response_object and final_response_object.candidates:
        logger.debug("Processing LLM final response candidates")
        part = final_response_object.candidates[0].content.parts[0]
        logger.debug(f"Response part type: {type(part)}")
        logger.debug(f"Part has function_call: {hasattr(part, 'function_call') and part.function_call}")
        logger.debug(f"Part has text: {hasattr(part, 'text') and bool(part.text)}")

        if part.function_call:
            logger.debug(f"Function call detected in final response: {part.function_call.name}")

            args = extract_function_args(part.function_call.args)

            return {
                "type": "function_call",
                "function": {
                    "name": part.function_call.name,
                    "args": args,
                    "id": str(uuid.uuid4())
                }
            }
        else:

            logger.warning("LLM did not call any function - this should not happen in Mode 2 only")
            return {
                "type": "function_call",
                "function": {
                    "name": "final_response",
                    "args": {"content": "I'm sorry, I couldn't generate a proper response."},
                    "id": str(uuid.uuid4())
                }
            }
    else:
        logger.warning("LLM final response was empty or failed")
        return {
            "type": "function_call",
            "function": {
                "name": "final_response",
                "args": {"content": "I'm sorry, I couldn't generate a final response."},
                "id": str(uuid.uuid4())
            }
        }


def process_message(msg: Dict[str, Any], llm_provider: LLMProvider) -> List[Dict[str, Any]]:
    logger.debug("=== Starting message processing ===")

    header = msg.get("header", {})
    payload = msg.get("payload", {})
    agent = payload.get("agent", {})
    context = agent.get("context", {})
    prompt_template = agent.get("prompt", "")

    logger.debug(f"Message header ID: {header.get('id', 'No ID')}")
    logger.debug(f"Context query: {context.get('query', 'No query')}")
    logger.debug(f"Context keys: {list(context.keys())}")
    logger.debug(f"Context history length: {len(context.get('history', []))}")
    logger.debug(f"Context interactions count: {len(context.get('interactions', []))}")
    logger.debug(f"Prompt template length: {len(prompt_template)} chars")

    template = Template(prompt_template)

    template_context = context.copy()
    if 'query' in template_context and 'user_query' not in template_context:
        template_context['user_query'] = template_context['query']

    if len(template_context.get('history', [])) > 0:
        logger.warning(f"Clearing {len(template_context['history'])} contaminated history items for testing")
        template_context['history'] = []

    prompt = template.safe_substitute(**template_context)
    logger.debug(f"Template context keys: {list(template_context.keys())}")

    if 'history' in template_context:
        logger.debug(f"History items: {template_context['history']}")

    logger.debug(f"Substituted prompt length: {len(prompt)} chars")

    logger.debug("Calling LLM provider...")

    print("=" * 80)
    print("FULL PROMPT BEING SENT TO GEMINI:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    print("END PROMPT")
    print("=" * 80)

    llm_response_object = llm_provider.generate(prompt)

    function_calls = []
    direct_response = ""
    llm_reasoning = ""

    if llm_response_object and llm_response_object.candidates:
        logger.debug("Processing LLM response candidates")
        part = llm_response_object.candidates[0].content.parts[0]
        logger.debug(f"Response part type: {type(part)}")
        logger.debug(f"Part has function_call: {hasattr(part, 'function_call') and part.function_call}")
        logger.debug(f"Part has text: {hasattr(part, 'text') and bool(part.text)}")

        if part.function_call:
            logger.debug(f"Function call detected: {part.function_call.name}")
            function_calls.append({
                "name": part.function_call.name,
                "args": extract_function_args(part.function_call.args)
            })
            llm_reasoning = f"Decided to call function: {part.function_call.name}"
            logger.debug(f"LLM reasoning: {llm_reasoning}")
            logger.debug(f"Function args: {extract_function_args(part.function_call.args)}")
        else:
            logger.debug("Direct response from LLM")
            direct_response = llm_response_object.text
            llm_reasoning = "Provided a direct answer."
            logger.debug(f"Direct response length: {len(direct_response)} chars")
            logger.debug(f"Direct response content: '{direct_response}'")
    else:
        logger.warning("LLM response was empty or failed")
        direct_response = "I'm having trouble connecting. Please try again."
        llm_reasoning = "LLM response was empty or failed."

    if "interactions" not in context:
        context["interactions"] = []
        logger.debug("Initialized empty interactions list")
    if "history" not in context:
        context["history"] = []
        logger.debug("Initialized empty history list")

    logger.debug(
        f"Current context has {len(context['interactions'])} interactions and {len(context['history'])} history items")

    response_entry = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "user_query": context.get("query", ""),
        "llm_reasoning": llm_reasoning,
        "function_executions": []
    }
    logger.debug(f"Created response entry with ID: {response_entry['interaction_id']}")

    messages = []

    if function_calls:
        logger.debug(f"Processing {len(function_calls)} function calls")
        first_func_call = function_calls[0]
        for i, func_call in enumerate(function_calls):
            execution_id = str(uuid.uuid4())
            logger.debug(f"Creating execution {i + 1}/{len(function_calls)} with ID: {execution_id}")

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
            logger.debug(f"Added to history: {tool_call_log}")

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
        logger.debug("Created function-assisted response message")

    else:
        logger.debug("Creating direct knowledge response")
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

    logger.debug(f"=== Message processing complete. Returning {len(messages)} messages ===")
    return messages


def process_function_response(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Process function call responses and update the structured hierarchy"""
    logger.debug("=== Starting function response processing ===")
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

        logger.debug(f"Context has {len(interactions)} interactions")

        function_result = function_call["response"]
        tool_response_log = f"Tool Response: {json.dumps(function_result)}"
        context["history"].append(tool_response_log)
        logger.debug(
            f"Added tool response to history (result keys: {list(function_result.keys()) if isinstance(function_result, dict) else 'non-dict result'})")

        execution_found = False
        for interaction in interactions:
            for execution in interaction.get("function_executions", []):
                if execution.get("execution_id") == func_call_id:
                    logger.debug(f"Found matching execution in interaction {interaction.get('interaction_id')}")
                    execution["execution_status"] = "completed"
                    execution["completed_at"] = time.time()
                    execution["execution_result"] = function_result
                    execution_found = True
                    break
            if execution_found:
                break

        if not execution_found:
            logger.warning(f"Could not find execution with ID {func_call_id} to update")

        latest_interaction = interactions[-1] if interactions else None
        all_completed = False

        if latest_interaction:
            executions = latest_interaction.get("function_executions", [])
            logger.debug(f"Latest interaction has {len(executions)} executions")

            completed_count = sum(1 for exec in executions if exec.get("execution_status") == "completed")
            logger.debug(f"Completed executions: {completed_count}/{len(executions)}")

            all_completed = all(
                exec.get("execution_status") == "completed"
                for exec in executions
            )
            logger.debug(f"All executions completed: {all_completed}")

            if all_completed:
                logger.debug("All function executions completed, creating summary")

                latest_interaction["execution_summary"] = {
                    "total_functions_executed": len(executions),
                    "all_successful": all(exec.get("error_details") is None for exec in executions),
                    "execution_results": [exec["execution_result"] for exec in executions if
                                          exec.get("execution_result")]
                }

                response_type = latest_interaction.get("response_type")
                logger.debug(f"Response type: {response_type}")

                if response_type == "function_assisted":
                    logger.info("All function executions completed, generating final synthesized response")

                    final_response_result = generate_final_response(context, agent.get("prompt", ""))

                    # MODE 1 REMOVED: Only function call routing is supported
                    if final_response_result.get("type") == "function_call":
                        logger.info(f"LLM requested function call: {final_response_result['function']['name']}")

                        # Create message structure for function routing
                        return {
                            "header": msg.get("header", {}),
                            "payload": {
                                "agent": {
                                    "context": context,
                                    "prompt": agent.get("prompt", ""),
                                    "current_function_execution": final_response_result["function"]
                                }
                            },
                            "route_to_function": True  # Signal for routing
                        }

                else:
                    logger.info("Direct knowledge query completed, no synthesis needed")

        context["interactions"] = interactions
        logger.debug("Updated context interactions")

        final_response = latest_interaction.get("response", "") if all_completed and latest_interaction else ""
        logger.debug(f"Final response length: {len(final_response)} chars")

        final_msg = {
            "header": msg.get("header", {}),
            "payload": {
                "agent": {
                    "context": context,
                    "prompt": agent.get("prompt", ""),
                    "response": final_response
                }
            }
        }
        logger.debug("=== Function response processing complete ===")
        return final_msg

    logger.debug("No response in function call, returning original message")
    return msg


def main():
    logger.info("Initializing LLM service application")

    app = Application(
        broker_address=os.getenv("KAFKA_BROKER", "localhost:9092"),
        consumer_group="llm-processor-group",
        auto_offset_reset="earliest"
    )
    logger.debug(f"Kafka broker: {os.getenv('KAFKA_BROKER', 'localhost:9092')}")

    llm_provider = GeminiProvider()

    request_topic = app.topic("agent-requests", value_deserializer="json")
    response_topic = app.topic("agent-function-responses", value_deserializer="json")
    logger.debug("Topics initialized: agent-requests, agent-function-responses")

    producer = app.get_producer()
    logger.debug("Producer initialized")

    def route_message(msg, target_topic):
        """Route message to specific topic"""
        header = msg.get("header", {})
        partition_key = header.get("id", "")

        logger.debug(f"Routing message (key: {partition_key}) to topic: {target_topic}")
        producer.produce(
            topic=target_topic,
            key=partition_key,
            value=json.dumps(msg)
        )
        logger.info(f"Message routed to topic: {target_topic}")

    def process_request(msg):
        """Process initial agent requests"""
        logger.info("=== Processing agent request ===")
        logger.debug(f"Request message header: {msg.get('header', {})}")

        messages = process_message(msg, llm_provider)
        logger.debug(f"Generated {len(messages)} messages from request")

        for i, message in enumerate(messages):
            function_call = message.get("payload", {}).get("agent", {}).get("current_function_execution")

            if function_call and "name" in function_call:
                target_topic = function_call["name"]
                logger.debug(f"Message {i + 1}: routing to function topic '{target_topic}'")
                route_message(message, target_topic)
            else:
                logger.debug(f"Message {i + 1}: routing to agent-responses (no function call)")
                route_message(message, "agent-responses")

    def process_response(msg):
        """Process function call responses - Updated to handle function routing"""
        logger.info("=== Processing function response ===")
        logger.debug(f"Response message header: {msg.get('header', {})}")

        result = process_function_response(msg)

        # Check if this needs to be routed to a function service
        if result.get("route_to_function"):
            function_execution = result.get("payload", {}).get("agent", {}).get("current_function_execution", {})
            function_name = function_execution.get("name")

            if function_name:
                logger.debug(f"Routing to function service: {function_name}")
                route_message(result, function_name)
            else:
                logger.warning("Function routing requested but no function name found")
                route_message(result, "agent-responses")
        else:
            logger.debug("Function response processed, routing to agent-responses")
            route_message(result, "agent-responses")

    request_stream = app.dataframe(request_topic)
    response_stream = app.dataframe(response_topic)

    request_stream.update(process_request)
    response_stream.update(process_response)
    logger.debug("Stream processors registered")

    logger.info("Starting LLM service...")
    app.run()


if __name__ == "__main__":
    main()


