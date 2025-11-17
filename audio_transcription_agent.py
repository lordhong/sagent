from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os
import tempfile
import requests
from pydub import AudioSegment

from .agents import AgentWithToolsAndMemory

class AudioTranscriptionAgent(AgentWithToolsAndMemory):
    """Agent for transcribing audio files using GPT-4's multimodal capabilities."""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[ConversationBufferMemory] = None,
        api_key: Optional[str] = None,
    ):
        # Initialize OpenAI
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter is required")
        
        # Initialize LLM if not provided
        if not llm:
            llm = ChatOpenAI(
                model="gpt-4-vision-preview",
                openai_api_key=api_key,
                max_tokens=4096
            )
        
        system_prompt = """You are an audio transcription agent specialized in converting speech to text using GPT-4.
        You can:
        - Transcribe audio files
        - Handle various audio formats
        - Support multiple languages
        - Clean and format transcriptions
        - Provide context and analysis of the audio content
        
        Always ensure the transcription is accurate and properly formatted."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=[],  # No additional tools needed for basic transcription
            memory=memory,
            name="Audio Transcription Agent",
            description="Transcribes audio files using GPT-4",
        )
    
    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using GPT-4's multimodal capabilities."""
        try:
            # Convert audio to a format GPT-4 can handle (if needed)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Load and convert audio if needed
                audio = AudioSegment.from_file(audio_path)
                audio.export(temp_path, format="mp3")
                
                # Read the audio file
                with open(temp_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                # Clean up
                os.unlink(temp_path)
            
            # Create messages for transcription
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": "Please transcribe this audio file. Include any relevant context or analysis."},
                    {"type": "audio", "audio": audio_data}
                ])
            ]
            
            # Get transcription
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        if input_text.startswith("transcribe:"):
            audio_path = input_text.split("transcribe:")[1].strip()
            return await self.transcribe_audio(audio_path)
        else:
            return await super().run(input_text) 