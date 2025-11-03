"""
Test refactored SupervisorAgent with model-driven agent selection
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agents.supervisor_agent_v2 import SupervisorAgent
from app.logger import logger


async def test_supervisor_decision():
    """Test Supervisor self-decision functionality"""
    
    supervisor = SupervisorAgent()
    
    print("\n" + "="*60)
    print("Testing Supervisor Model-Driven Decision Making")
    print("="*60)
    
    test_cases = [
        {
            "message": "Please analyze this remote sensing image and identify buildings and roads",
            "file_list": None,
            "expected_agent": "VisionAgent"
        },
        {
            "message": "Generate a detailed analysis report including image description and object detection",
            "file_list": None,
            "expected_agent": "DocAgent"
        },
        {
            "message": "Hello, how are you today?",
            "file_list": None,
            "expected_agent": "ChatAgent"
        },
        {
            "message": "I need to write documentation about image processing techniques",
            "file_list": None,
            "expected_agent": "DocAgent"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\nTest {i}: {test_case['message']}")
            print(f"Expected Agent: {test_case['expected_agent']}")
            
            response = await supervisor.chat_response(
                message=test_case['message'],
                file_list=test_case['file_list']
            )
            
            selected_agent = response.get("next_agent", "Unknown")
            reasoning = response.get("planning_reasoning", "None")
            
            print(f"Selected Agent: {selected_agent}")
            print(f"Decision Reasoning: {reasoning}")
            
            if selected_agent == test_case['expected_agent']:
                print("Result matches expectation!")
            else:
                print(f"Note: Got {selected_agent}, expected {test_case['expected_agent']}")
                
        except Exception as e:
            print(f"Error in test {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


async def test_valid_agent_names():
    """Test Agent list retrieval"""
    print("\n" + "="*60)
    print("Testing Available Agents")
    print("="*60)
    
    supervisor = SupervisorAgent()
    agents = supervisor.get_agents()
    print(f"\nAvailable Agents: {agents}")
    print(f"Total Agents: {len(agents)}")
    
    print("\nAgent Details:")
    for agent_info in supervisor.agent_infos:
        print(f"  - {agent_info['name']}: {agent_info['description']}")


if __name__ == "__main__":
    print("\nStarting SupervisorAgent Refactored Tests\n")
    
    asyncio.run(test_valid_agent_names())
    asyncio.run(test_supervisor_decision())
    
    print("\nTests completed!")
