from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
from langchain.tools import tool
import intercom
import os
import json
from datetime import datetime

from .agents import AgentWithToolsAndMemory

class ChatAgent(AgentWithToolsAndMemory):
    """Agent for handling chat conversations using Intercom's Messages API."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[ConversationBufferMemory] = None,
        api_key: Optional[str] = None,
        admin_id: Optional[str] = None,
    ):
        # Initialize Intercom
        api_key = api_key or os.getenv("INTERCOM_API_KEY")
        admin_id = admin_id or os.getenv("INTERCOM_ADMIN_ID")
        
        if not api_key or not admin_id:
            raise ValueError("INTERCOM_API_KEY and INTERCOM_ADMIN_ID environment variables or parameters are required")
        
        intercom.api_key = api_key
        self.admin_id = admin_id
        
        # Create Intercom tools
        tools: List[BaseTool] = [
            self._create_send_message_tool(),
            self._create_list_conversations_tool(),
            self._create_get_conversation_tool(),
            self._create_reply_to_conversation_tool()
        ]
        
        system_prompt = """You are a chat agent specialized in handling conversations using Intercom.
        You can:
        - Send messages to users
        - List and manage conversations
        - Reply to existing conversations
        - Provide customer support
        - Handle multiple conversations simultaneously
        - Track conversation history
        
        Always ensure:
        - Messages are clear and professional
        - Responses are timely and helpful
        - Customer context is maintained
        - Sensitive information is handled securely
        - Conversations are properly tagged and organized"""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="Chat Agent",
            description="Handles chat conversations using Intercom",
        )
    
    def _create_send_message_tool(self) -> BaseTool:
        @tool
        def send_message(
            user_id: str,
            message: str,
            message_type: str = "inapp",
            tags: Optional[List[str]] = None
        ) -> str:
            """Send a message to a user."""
            try:
                # Create message
                message_data = {
                    "message_type": message_type,
                    "body": message,
                    "from": {
                        "type": "admin",
                        "id": self.admin_id
                    },
                    "to": {
                        "type": "user",
                        "id": user_id
                    }
                }
                
                if tags:
                    message_data["tags"] = tags
                
                response = intercom.messages.create(**message_data)
                
                return json.dumps({
                    "status": "success",
                    "message_id": response.id,
                    "sent_at": datetime.now().isoformat()
                })
            except intercom.errors.IntercomError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return send_message
    
    def _create_list_conversations_tool(self) -> BaseTool:
        @tool
        def list_conversations(
            user_id: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 10
        ) -> str:
            """List conversations, optionally filtered by user or status."""
            try:
                params = {"limit": limit}
                if user_id:
                    params["user_id"] = user_id
                if status:
                    params["status"] = status
                
                conversations = intercom.conversations.list(**params)
                
                return json.dumps({
                    "status": "success",
                    "conversations": [
                        {
                            "id": conv.id,
                            "user_id": conv.user.id,
                            "status": conv.status,
                            "created_at": conv.created_at,
                            "updated_at": conv.updated_at
                        }
                        for conv in conversations
                    ]
                })
            except intercom.errors.IntercomError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return list_conversations
    
    def _create_get_conversation_tool(self) -> BaseTool:
        @tool
        def get_conversation(conversation_id: str) -> str:
            """Get details of a specific conversation."""
            try:
                conversation = intercom.conversations.find(id=conversation_id)
                
                return json.dumps({
                    "status": "success",
                    "conversation": {
                        "id": conversation.id,
                        "user_id": conversation.user.id,
                        "status": conversation.status,
                        "messages": [
                            {
                                "id": msg.id,
                                "body": msg.body,
                                "author": msg.author.type,
                                "created_at": msg.created_at
                            }
                            for msg in conversation.messages
                        ],
                        "created_at": conversation.created_at,
                        "updated_at": conversation.updated_at
                    }
                })
            except intercom.errors.IntercomError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return get_conversation
    
    def _create_reply_to_conversation_tool(self) -> BaseTool:
        @tool
        def reply_to_conversation(
            conversation_id: str,
            message: str,
            message_type: str = "inapp"
        ) -> str:
            """Reply to an existing conversation."""
            try:
                response = intercom.conversations.reply(
                    id=conversation_id,
                    message_type=message_type,
                    body=message,
                    admin_id=self.admin_id
                )
                
                return json.dumps({
                    "status": "success",
                    "message_id": response.id,
                    "conversation_id": conversation_id,
                    "sent_at": datetime.now().isoformat()
                })
            except intercom.errors.IntercomError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return reply_to_conversation
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        # The agent will use its tools to handle chat conversations
        return await super().run(input_text) 