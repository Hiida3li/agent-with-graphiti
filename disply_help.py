def display_workflow_info(self, payload):
    """Display workflow information"""
    context = payload.get("payload", {}).get("agent", {}).get("context", {})
    history = context.get("history", [])

    if history:
        latest = history[-1]
        print(f" Latest Interaction:")
        if isinstance(latest, dict):
            print(f"   Query: {latest.get('user_query', 'N/A')}")
            print(f"   Response Type: {latest.get('response_type', 'N/A')}")

            # Only try to get executions if latest is a dict
            executions = latest.get("function_executions", [])
            if executions:
                print(f"   Function Executions: {len(executions)}")
                for i, exec in enumerate(executions, 1):
                    status = exec.get("execution_status", "unknown")
                    name = exec.get("function_name", "unknown")
                    print(f"     {i}. {name}: {status}")

        elif isinstance(latest, str):
            print(f"   Query: {latest}")
            print(f"   Response Type: direct_knowledge")
        else:
            print(f"   Query: Unknown format")
            print(f"   Response Type: unknown")
        print()


def display_help(self):
    """Display help information"""
    print("""
🤖 LLM Agent Chat CLI Commands:

  <message>     - Send a message to the agent
  /help         - Show this help
  /status       - Show current payload status
  /history      - Show conversation history
  /files        - List saved payload files
  /clear        - Clear conversation history
  /check        - Check for new responses manually
  /quit         - Exit the chat
        """)


