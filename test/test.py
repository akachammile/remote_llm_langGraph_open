from typing import List, TypedDict
from langgraph.graph import StateGraph, END

# 1. Define the tool tree structure
# The keys are the names of the tools/tool categories.
# 'description' is a brief explanation of what the tool/category does.
# 'children' contains sub-tools or sub-categories. Leaf nodes do not have 'children'.
tool_tree = {
    "search": {
        "description": "Tools for searching",
        "children": {
            "web_search": {"description": "Search the web for information."},
            "code_search": {"description": "Search within the codebase."},
        },
    },
    "file_system": {
        "description": "Tools for file system operations",
        "children": {
            "list_files": {"description": "List files in a specified directory."},
            "read_file": {"description": "Read the contents of a file."},
            "write_file": {"description": "Write content to a file."},
        },
    },
    "communication": {
        "description": "Tools for communication",
        "children": {
            "send_email": {"description": "Send an email to a recipient."},
            "send_slack_message": {"description": "Send a message to a Slack channel."},
        },
    },
    "development": {
        "description": "Tools for software development",
        "children": {
            "run_tests": {"description": "Run automated tests."},
            "run_linter": {"description": "Run a linter to check code quality."},
            "get_git_diff": {"description": "Get the diff of the current git branch."},
        },
    },
}

# 2. Define the state for the LangGraph graph
class AgentState(TypedDict):
    # The user's query
    query: str
    # The current path in the tool tree
    path: List[str]
    # The result of the last operation, used for routing
    result: str

# 3. Mock LLM for routing decisions
def mock_llm_router(query: str, options: List[str]) -> str:
    """
    A mock LLM to simulate choosing the best tool/category.
    In a real application, you would replace this with a call to an actual LLM.
    """
    print(f"LLM Router: Choosing from {options} for query '{query}'")
    # Simple heuristic: if a tool name is in the query, choose it.
    for option in options:
        if option in query:
            print(f"LLM chose: {option}")
            return option
    # Otherwise, just pick the first one for this demo.
    if options:
        print(f"LLM chose: {options[0]}")
        return options[0]
    # If no options, signal to backtrack.
    return "backtrack"

# 4. Define the nodes for the graph

def router_node(state: AgentState):
    """
    This node decides which path to take next.
    It navigates the tool_tree based on the current path.
    """
    print("--- Router Node ---")
    current_node = tool_tree
    # Traverse the tree to the current node
    for key in state["path"]:
        current_node = current_node[key]["children"]

    options = list(current_node.keys())
    print(f"Current path: {state['path']}")
    print(f"Available options: {options}")

    # If we are at a leaf node (a tool), execute it.
    is_leaf = "children" not in next(iter(current_node.values()), {})
    if is_leaf:
        return {"result": "tool_execution"}

    # Use the mock LLM to choose the next step
    choice = mock_llm_router(state["query"], options)
    
    if choice == "backtrack":
        # If we can't make a choice, backtrack.
        if not state["path"]:
            # If we are at the root and can't decide, end.
            return {"result": "end"}
        return {"result": "backtrack"}

    # Continue down the chosen path
    new_path = state["path"] + [choice]
    return {"path": new_path, "result": "continue"}

def tool_node(state: AgentState):
    """
    This node simulates the execution of a tool.
    """
    tool_name = state["path"][-1]
    print(f"--- Tool Node: {tool_name} ---")
    # In a real implementation, you would execute the actual tool here.
    print(f"Executing tool '{tool_name}' with query '{state['query']}'")
    return {"result": f"Successfully executed {tool_name}"}

def backtrack_node(state: AgentState):
    """
    This node handles backtracking by moving up one level in the tree.
    """
    print("--- Backtracking ---")
    new_path = state["path"][:-1]
    # We remove the last choice and will try again from the parent node.
    # A more sophisticated implementation could mark paths as "already tried".
    return {"path": new_path, "result": "continue"}

# 5. Define the graph and its edges
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("tool_executor", tool_node)
workflow.add_node("backtrack", backtrack_node)

workflow.set_entry_point("router")

def should_continue(state: AgentState):
    """
    Conditional logic to route between nodes.
    """
    if state["result"] == "end":
        return END
    if state["result"] == "tool_execution":
        return "tool_executor"
    if state["result"] == "backtrack":
        return "backtrack"
    return "router"

# Add conditional edges based on the 'result' field in the state
workflow.add_conditional_edges("router", should_continue)
# After executing a tool, the process ends.
workflow.add_edge("tool_executor", END)
# After backtracking, we go back to the router to make a new decision.
workflow.add_edge("backtrack", "router")

# Compile the graph into a runnable application
app = workflow.compile()

# 6. Run the demo with a few example queries
def run_demo(query):
    print(f"\n--- Running demo for query: '{query}' ---")
    initial_state = {"query": query, "path": [], "result": ""}
    final_state = app.invoke(initial_state)
    print(f"Final result: {final_state.get('result')}")

if __name__ == "__main__":
    run_demo("I need to search the web for the latest news.")
    run_demo("Can you list the files in the current directory?")
    run_demo("I want to send an email to my team.")
    run_demo("Let's check for git changes.")
    # This query doesn't match any tool, so it will try a path and then backtrack.
    run_demo("I have no idea what to do, maybe something with development.")