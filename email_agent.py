from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.gmail import (
    GmailCreateDraft,
    GmailGetMessage,
    GmailGetThread,
    GmailSearch,
    GmailSendMessage,
)
from langchain_community.tools.gmail.utils import build_resource_service
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle

from .agents import AgentWithToolsAndMemory

class EmailAgent(AgentWithToolsAndMemory):
    """Agent for handling email-related tasks using Gmail."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[ConversationBufferMemory] = None,
        credentials_path: str = "credentials.json",
        token_path: str = "token.pickle",
    ):
        # Initialize Gmail service
        self.service = self._get_gmail_service(credentials_path, token_path)
        
        # Create Gmail tools
        tools = [
            GmailSearch(api_resource=self.service),
            GmailGetMessage(api_resource=self.service),
            GmailGetThread(api_resource=self.service),
            GmailCreateDraft(api_resource=self.service),
            GmailSendMessage(api_resource=self.service),
        ]
        
        system_prompt = """You are an email agent specialized in handling email-related tasks using Gmail.
        You can:
        - Search emails using GmailSearch
        - Read specific messages using GmailGetMessage
        - Read entire threads using GmailGetThread
        - Create email drafts using GmailCreateDraft
        - Send emails using GmailSendMessage
        
        Always be professional and clear in your email communications.
        When searching emails, use appropriate search queries.
        When creating or sending emails, ensure all necessary fields are properly filled."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="Email Agent",
            description="Handles email-related tasks using Gmail",
        )
    
    def _get_gmail_service(self, credentials_path: str, token_path: str):
        """Get Gmail service using OAuth2 credentials."""
        SCOPES = [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.modify",
        ]
        
        creds = None
        if os.path.exists(token_path):
            with open(token_path, "rb") as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(token_path, "wb") as token:
                pickle.dump(creds, token)
        
        return build_resource_service(credentials=creds) 