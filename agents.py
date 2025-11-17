from typing import Any, Dict, List, Optional, Type, Union
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain.tools import tool
import os
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
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
import pickle

from .email_agent import EmailAgent
from .image_ocr_agent import ImageOCRAgent
from .audio_transcription_agent import AudioTranscriptionAgent
from .video_transcription_agent import VideoTranscriptionAgent
from .sms_agent import SMSAgent
from .bank_submission_agent import BankSubmissionAgent
from .chat_agent import ChatAgent

class Agent:
    """Base class for all agents."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        name: str = "Agent",
        description: str = "A general-purpose agent",
        tools: Optional[List[Any]] = None,
        memory: Optional[Any] = None,
        knowledge: Optional[Any] = None,
        planning: Optional[Any] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.name = name
        self.description = description
        self.tools = tools or []
        self.memory = memory
        self.knowledge = knowledge
        self.planning = planning
        
        # Create the base chain
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """Create the base chain for the agent."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        
        return prompt | self.llm | StrOutputParser()
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        return await self.chain.ainvoke({"input": input_text})
    
    def get_agent_type(self) -> str:
        """Get the type of the agent."""
        return self.__class__.__name__
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.get_agent_type(),
            "has_tools": bool(self.tools),
            "has_memory": bool(self.memory),
            "has_knowledge": bool(self.knowledge),
            "has_planning": bool(self.planning),
        }

class AgentWithTools(Agent):
    """Agent with tools capability."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        tools: List[BaseTool],
        name: str = "AgentWithTools",
        description: str = "An agent with tools capability",
    ):
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            name=name,
            description=description,
            tools=tools,
        )
        
        # Create the agent with tools
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create an agent executor with tools."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
        )
    
    async def run(self, input_text: str) -> str:
        """Run the agent with tools."""
        result = await self.agent_executor.ainvoke({"input": input_text})
        return result["output"]

class AgentWithMemory(Agent):
    """Agent with memory capability."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        memory: Optional[ConversationBufferMemory] = None,
        name: str = "AgentWithMemory",
        description: str = "An agent with memory capability",
    ):
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            name=name,
            description=description,
            memory=memory,
        )
        
        # Create the chain with memory
        self.chain = self._create_chain_with_memory()
    
    def _create_chain_with_memory(self):
        """Create a chain with memory."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        return {
            "input": lambda x: x["input"],
            "chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"],
        } | chain
    
    async def run(self, input_text: str) -> str:
        """Run the agent with memory."""
        result = await self.chain.ainvoke({"input": input_text})
        await self.memory.save_context({"input": input_text}, {"output": result})
        return result

class AgentWithToolsAndMemory(Agent):
    """Agent with both tools and memory capabilities."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        tools: List[BaseTool],
        memory: Optional[ConversationBufferMemory] = None,
        name: str = "AgentWithToolsAndMemory",
        description: str = "An agent with both tools and memory capabilities",
    ):
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            name=name,
            description=description,
            tools=tools,
            memory=memory,
        )
        
        # Create the agent executor with tools and memory
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create an agent executor with tools and memory."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
        )
    
    async def run(self, input_text: str) -> str:
        """Run the agent with tools and memory."""
        result = await self.agent_executor.ainvoke({"input": input_text})
        return result["output"]

class AgentWithToolsAndMemoryAndKnowledge(Agent):
    """Agent with tools, memory, and knowledge capabilities."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        tools: List[BaseTool],
        memory: Optional[ConversationBufferMemory] = None,
        knowledge: Optional[VectorStore] = None,
        name: str = "AgentWithToolsAndMemoryAndKnowledge",
        description: str = "An agent with tools, memory, and knowledge capabilities",
    ):
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        
        if knowledge is None:
            # Create a default knowledge base with empty documents
            embeddings = OpenAIEmbeddings()
            knowledge = VectorStore.from_documents(
                documents=[],
                embedding=embeddings,
            )
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            name=name,
            description=description,
            tools=tools,
            memory=memory,
            knowledge=knowledge,
        )
        
        # Create the agent executor with tools, memory, and knowledge
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create an agent executor with tools, memory, and knowledge."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
        )
    
    async def run(self, input_text: str) -> str:
        """Run the agent with tools, memory, and knowledge."""
        # First, search the knowledge base for relevant information
        relevant_docs = await self.knowledge.asimilarity_search(input_text, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Add the context to the input
        augmented_input = f"Context from knowledge base:\n{context}\n\nUser input: {input_text}"
        
        # Run the agent with the augmented input
        result = await self.agent_executor.ainvoke({"input": augmented_input})
        return result["output"]
    
    async def add_to_knowledge(self, text: str) -> None:
        """Add text to the knowledge base."""
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(text)
        
        # Add the chunks to the knowledge base
        await self.knowledge.aadd_texts(chunks)

class AgentWithToolsAndMemoryAndKnowledgeAndPlanning(Agent):
    """Agent with tools, memory, knowledge, and planning capabilities."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        tools: List[BaseTool],
        memory: Optional[ConversationBufferMemory] = None,
        knowledge: Optional[VectorStore] = None,
        name: str = "AgentWithToolsAndMemoryAndKnowledgeAndPlanning",
        description: str = "An agent with tools, memory, knowledge, and planning capabilities",
    ):
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
            )
        
        if knowledge is None:
            # Create a default knowledge base with empty documents
            embeddings = OpenAIEmbeddings()
            knowledge = VectorStore.from_documents(
                documents=[],
                embedding=embeddings,
            )
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            name=name,
            description=description,
            tools=tools,
            memory=memory,
            knowledge=knowledge,
            planning=True,
        )
        
        # Create the plan and execute agent
        self.plan_and_execute = self._create_plan_and_execute()
    
    def _create_plan_and_execute(self) -> PlanAndExecute:
        """Create a plan and execute agent."""
        # Create the planner
        planner = load_chat_planner(
            llm=self.llm,
            system_prompt=self.system_prompt,
        )
        
        # Create the executor
        executor = load_agent_executor(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
        
        return PlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=True,
        )
    
    async def run(self, input_text: str) -> str:
        """Run the agent with tools, memory, knowledge, and planning."""
        # First, search the knowledge base for relevant information
        relevant_docs = await self.knowledge.asimilarity_search(input_text, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Add the context to the input
        augmented_input = f"Context from knowledge base:\n{context}\n\nUser input: {input_text}"
        
        # Run the plan and execute agent
        result = await self.plan_and_execute.ainvoke({"input": augmented_input})
        return result["output"]
    
    async def add_to_knowledge(self, text: str) -> None:
        """Add text to the knowledge base."""
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(text)
        
        # Add the chunks to the knowledge base
        await self.knowledge.aadd_texts(chunks)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str

def create_workflow(
    orchestrator: Any,
    email_agent: Any,
    image_ocr_agent: Any,
    audio_transcription_agent: Any,
    video_transcription_agent: Any,
    sms_agent: Any,
    chat_agent: Any,
    evaluator: Any,
    bank_submission: Any,
) -> StateGraph:
    """Create the LangGraph workflow for agent orchestration."""
    
    # Define the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("Orchestrator", orchestrator)
    workflow.add_node("Email Agent", email_agent)
    workflow.add_node("Image OCR Agent", image_ocr_agent)
    workflow.add_node("Audio Transcription Agent", audio_transcription_agent)
    workflow.add_node("Video Transcription Agent", video_transcription_agent)
    workflow.add_node("SMS Agent", sms_agent)
    workflow.add_node("Chat Agent", chat_agent)
    workflow.add_node("Evaluator", evaluator)
    workflow.add_node("Bank Submission", bank_submission)
    
    # Set entry point
    workflow.set_entry_point("Orchestrator")
    
    # Add edges from Orchestrator to agents
    workflow.add_edge("Orchestrator", "Email Agent")
    workflow.add_edge("Orchestrator", "Image OCR Agent")
    workflow.add_edge("Orchestrator", "Audio Transcription Agent")
    workflow.add_edge("Orchestrator", "Video Transcription Agent")
    workflow.add_edge("Orchestrator", "SMS Agent")
    workflow.add_edge("Orchestrator", "Chat Agent")
    
    # Add edges from agents to Evaluator
    workflow.add_edge("Email Agent", "Evaluator")
    workflow.add_edge("Image OCR Agent", "Evaluator")
    workflow.add_edge("Audio Transcription Agent", "Evaluator")
    workflow.add_edge("Video Transcription Agent", "Evaluator")
    workflow.add_edge("SMS Agent", "Evaluator")
    workflow.add_edge("Chat Agent", "Evaluator")
    
    # Add conditional edges from Evaluator
    def should_continue(state: AgentState) -> str:
        """Determine if the workflow should continue or end."""
        return state.get("next", "continue")
    
    workflow.add_conditional_edges(
        "Evaluator",
        should_continue,
        {
            "continue": "Orchestrator",
            "end": "Bank Submission",
        },
    )
    
    # Add edge from Bank Submission to END
    workflow.add_edge("Bank Submission", END)
    
    return workflow.compile()

class OrchestratorAgent(AgentWithToolsAndMemory):
    """Agent responsible for orchestrating the workflow."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        memory: Optional[ConversationBufferMemory] = None,
    ):
        system_prompt = """You are an orchestrator agent responsible for routing tasks to the appropriate specialized agents.
        Based on the user's request, determine which agent should handle it:
        - Email Agent: For email-related tasks
        - Image OCR Agent: For image text extraction
        - Audio Transcription Agent: For audio file transcription
        - Video Transcription Agent: For video file transcription
        - SMS Agent: For SMS-related tasks
        - Chat Agent: For general conversation
        
        Always respond with the name of the appropriate agent."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="Orchestrator",
            description="Routes tasks to appropriate specialized agents",
        )

class EvaluatorAgent(AgentWithToolsAndMemory):
    """Agent for evaluating task completion and determining next steps."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        memory: Optional[ConversationBufferMemory] = None,
    ):
        system_prompt = """You are an evaluator agent responsible for determining if tasks are complete.
        You should:
        - Evaluate if the current task is complete
        - Determine if additional steps are needed
        - Decide whether to continue or end the workflow
        
        Respond with either 'continue' if more work is needed or 'end' if the task is complete."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="Evaluator",
            description="Evaluates task completion and determines next steps",
        )
        
        # Initialize ONNX runtime session
        model_path = Path("models/win_rate_prediction_model.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.session = ort.InferenceSession(str(model_path))
    
    def _prepare_input(self, state: AgentState) -> np.ndarray:
        """Prepare input features for the model."""
        # Extract relevant features from the state
        # This is a placeholder - you'll need to adjust based on your actual model's input requirements
        features = {
            'feature1': 0.5,  # Replace with actual feature extraction
            'feature2': 0.3,  # Replace with actual feature extraction
            # Add more features as needed
        }
        
        # Convert to numpy array
        input_data = np.array([list(features.values())], dtype=np.float32)
        return input_data
    
    async def run(self, input_text: str) -> str:
        """Run the evaluator with win rate prediction."""
        # Prepare input for the model
        input_data = self._prepare_input(self.memory.load_memory_variables({}))
        
        # Run prediction
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        win_rate = self.session.run([output_name], {input_name: input_data})[0][0][0]
        
        # Make decision based on win rate
        if win_rate >= 0.5:
            return "end"  # Go to final submission
        else:
            return "continue"  # Go back to orchestrator

class BankSubmissionAgent(AgentWithToolsAndMemory):
    """Agent for handling bank submission tasks."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        memory: Optional[ConversationBufferMemory] = None,
    ):
        system_prompt = """You are a bank submission agent responsible for finalizing and submitting tasks.
        You should:
        - Review all completed work
        - Ensure all requirements are met
        - Format the final submission
        - Handle any final processing
        
        Always ensure the submission is complete and properly formatted."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            memory=memory,
            name="Bank Submission",
            description="Handles final submission of completed tasks",
        ) 