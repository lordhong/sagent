from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import google.generativeai as genai
import os

from .agents import AgentWithToolsAndMemory

class ImageOCRAgent(AgentWithToolsAndMemory):
    """Agent for extracting text from images using Google Gemini."""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[ConversationBufferMemory] = None,
        api_key: Optional[str] = None,
    ):
        # Initialize Gemini
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or api_key parameter is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')
        
        # Initialize LLM if not provided
        if not llm:
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                convert_system_message_to_human=True
            )
        
        system_prompt = """You are an image OCR agent specialized in extracting text from images using Google Gemini.
        You can:
        - Process images for text extraction
        - Handle various image formats
        - Clean and format extracted text
        - Handle multiple languages
        - Provide context about the extracted text
        
        Always ensure the extracted text is accurate and properly formatted."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=[],  # No additional tools needed for basic OCR
            memory=memory,
            name="Image OCR Agent",
            description="Extracts text from images using Google Gemini",
        )
    
    async def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using Gemini."""
        try:
            # Load the image
            image = genai.types.Image.load_from_file(image_path)
            
            # Generate content
            response = self.model.generate_content([
                "Extract all text from this image. If there are multiple languages, identify them. Format the output clearly.",
                image
            ])
            
            return response.text
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        if input_text.startswith("extract:"):
            image_path = input_text.split("extract:")[1].strip()
            return await self.extract_text_from_image(image_path)
        else:
            return await super().run(input_text) 