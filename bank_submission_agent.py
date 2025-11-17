from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import BaseTool
from langchain.tools import tool
import stripe
import os
import json
from datetime import datetime

from .agents import AgentWithToolsAndMemory

class BankSubmissionAgent(AgentWithToolsAndMemory):
    """Agent for handling bank submission tasks using Stripe's dispute submission API."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[ConversationBufferMemory] = None,
        api_key: Optional[str] = None,
    ):
        # Initialize Stripe
        api_key = api_key or os.getenv("STRIPE_API_KEY")
        if not api_key:
            raise ValueError("STRIPE_API_KEY environment variable or api_key parameter is required")
        
        stripe.api_key = api_key
        
        # Create Stripe tools
        tools: List[BaseTool] = [
            self._create_dispute_submission_tool(),
            self._create_evidence_upload_tool(),
            self._create_dispute_status_tool()
        ]
        
        system_prompt = """You are a bank submission agent specialized in handling dispute submissions using Stripe.
        You can:
        - Submit disputes to banks
        - Upload evidence for disputes
        - Check dispute status
        - Provide guidance on dispute resolution
        - Analyze dispute patterns
        - Suggest preventive measures
        
        Always ensure:
        - All required evidence is properly documented
        - Submissions follow bank guidelines
        - Evidence is clear and compelling
        - Timelines are adhered to
        - Communication is professional and clear"""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="Bank Submission Agent",
            description="Handles bank dispute submissions using Stripe",
        )
    
    def _create_dispute_submission_tool(self) -> BaseTool:
        @tool
        def submit_dispute(
            dispute_id: str,
            evidence: Dict[str, Any],
            reason: str
        ) -> str:
            """Submit a dispute to the bank with evidence."""
            try:
                dispute = stripe.Dispute.retrieve(dispute_id)
                
                # Submit evidence
                dispute = stripe.Dispute.modify(
                    dispute_id,
                    evidence=evidence,
                    reason=reason
                )
                
                return json.dumps({
                    "status": "success",
                    "dispute_id": dispute.id,
                    "status": dispute.status,
                    "submitted_at": datetime.now().isoformat()
                })
            except stripe.error.StripeError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return submit_dispute
    
    def _create_evidence_upload_tool(self) -> BaseTool:
        @tool
        def upload_evidence(
            dispute_id: str,
            evidence_type: str,
            evidence_data: Dict[str, Any]
        ) -> str:
            """Upload evidence for a dispute."""
            try:
                dispute = stripe.Dispute.retrieve(dispute_id)
                
                # Upload evidence
                dispute = stripe.Dispute.modify(
                    dispute_id,
                    evidence={
                        evidence_type: evidence_data
                    }
                )
                
                return json.dumps({
                    "status": "success",
                    "dispute_id": dispute.id,
                    "evidence_type": evidence_type,
                    "uploaded_at": datetime.now().isoformat()
                })
            except stripe.error.StripeError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return upload_evidence
    
    def _create_dispute_status_tool(self) -> BaseTool:
        @tool
        def check_dispute_status(dispute_id: str) -> str:
            """Check the status of a dispute."""
            try:
                dispute = stripe.Dispute.retrieve(dispute_id)
                
                return json.dumps({
                    "status": "success",
                    "dispute_id": dispute.id,
                    "current_status": dispute.status,
                    "amount": dispute.amount,
                    "currency": dispute.currency,
                    "reason": dispute.reason,
                    "evidence_due_by": dispute.evidence_due_by
                })
            except stripe.error.StripeError as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e)
                })
        
        return check_dispute_status
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        # The agent will use its tools to handle bank submission tasks
        return await super().run(input_text) 