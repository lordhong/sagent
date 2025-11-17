from typing import Optional, List
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import TwilioTool
from langchain_core.tools import BaseTool
import os

from .agents import AgentWithToolsAndMemory

class SMSAgent(AgentWithToolsAndMemory):
    """Agent for handling SMS-related tasks using Twilio."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[ConversationBufferMemory] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
    ):
        # Get Twilio credentials
        account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
        
        if not all([account_sid, auth_token, from_number]):
            raise ValueError("TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER environment variables or parameters are required")
        
        # Create Twilio tools
        tools: List[BaseTool] = [
            TwilioTool(
                account_sid=account_sid,
                auth_token=auth_token,
                from_number=from_number,
                name="send_sms",
                description="Send an SMS message to a phone number"
            ),
            TwilioTool(
                account_sid=account_sid,
                auth_token=auth_token,
                from_number=from_number,
                name="list_messages",
                description="List SMS messages from a specific phone number"
            ),
            TwilioTool(
                account_sid=account_sid,
                auth_token=auth_token,
                from_number=from_number,
                name="get_message",
                description="Get details of a specific SMS message"
            )
        ]
        
        system_prompt = """You are an SMS agent specialized in handling SMS-related tasks using Twilio.
        You can:
        - Send SMS messages to phone numbers
        - List SMS messages from specific numbers
        - Get details of specific messages
        - Manage SMS conversations
        
        Always be concise and clear in your SMS communications.
        When sending messages, ensure the content is appropriate and properly formatted.
        When retrieving messages, provide relevant context and details."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="SMS Agent",
            description="Handles SMS-related tasks using Twilio",
        )
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        # The agent will use its tools to handle SMS-related tasks
        return await super().run(input_text) 