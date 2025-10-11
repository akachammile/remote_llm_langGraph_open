from app.graphs.graph_state import AgentState

class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """初始化逻辑器"""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_segment(self, state: AgentState):
        """判断是否要继续分割"""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_detect(self, state: AgentState):
        """是否要继续检测"""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """是否检测变化."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    # def should_continue_fundamentals(self, state: AgentState):
    #     """Determine if fundamentals analysis should continue."""
    #     messages = state["messages"]
    #     last_message = messages[-1]
    #     if last_message.tool_calls:
    #         return "tools_fundamentals"
    #     return "Msg Clear Fundamentals"

    # def should_continue_debate(self, state: AgentState) -> str:
    #     """Determine if debate should continue."""

    #     if (
    #         state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
    #     ):  # 3 rounds of back-and-forth between 2 agents
    #         return "Research Manager"
    #     if state["investment_debate_state"]["current_response"].startswith("Bull"):
    #         return "Bear Researcher"
    #     return "Bull Researcher"

    # def should_continue_risk_analysis(self, state: AgentState) -> str:
    #     """Determine if risk analysis should continue."""
    #     if (
    #         state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
    #     ):  # 3 rounds of back-and-forth between 3 agents
    #         return "Risk Judge"
    #     if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
    #         return "Safe Analyst"
    #     if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
    #         return "Neutral Analyst"
    #     return "Risky Analyst"
