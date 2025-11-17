from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import os
import tempfile
import requests
from pydub import AudioSegment
import cv2
import numpy as np
import base64

from .agents import AgentWithToolsAndMemory

class VideoTranscriptionAgent(AgentWithToolsAndMemory):
    """Agent for transcribing video files using GPT-4's multimodal capabilities."""
    
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
        
        system_prompt = """You are a video transcription agent specialized in converting video content to text using GPT-4.
        You can:
        - Transcribe video files
        - Handle various video formats
        - Support multiple languages
        - Clean and format transcriptions
        - Handle video subtitles
        - Provide context and analysis of the video content
        - Describe visual elements when relevant
        
        Always ensure the transcription is accurate and properly formatted."""
        
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            tools=[],  # No additional tools needed for basic transcription
            memory=memory,
            name="Video Transcription Agent",
            description="Transcribes video files using GPT-4",
        )
    
    def _extract_key_frames(self, video_path: str, num_frames: int = 5) -> list:
        """Extract key frames from video."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval
        interval = max(1, total_frames // num_frames)
        
        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        return frames
    
    async def transcribe_video(self, video_path: str) -> str:
        """Transcribe video using GPT-4's multimodal capabilities."""
        try:
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
                # Load and convert audio
                video = AudioSegment.from_file(video_path)
                video.export(temp_audio_path, format="mp3")
                
                # Read the audio file
                with open(temp_audio_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                # Clean up audio file
                os.unlink(temp_audio_path)
            
            # Extract key frames
            frames = self._extract_key_frames(video_path)
            
            # Prepare frame data
            frame_data = []
            for frame in frames:
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frame_data.append(frame_base64)
            
            # Create messages for transcription
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": "Please transcribe this video file. Include any relevant context, analysis, and visual descriptions."},
                    {"type": "audio", "audio": audio_data},
                    *[{"type": "image", "image": frame} for frame in frame_data]
                ])
            ]
            
            # Get transcription
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error processing video: {str(e)}"
    
    async def run(self, input_text: str) -> str:
        """Run the agent with the given input."""
        if input_text.startswith("transcribe:"):
            video_path = input_text.split("transcribe:")[1].strip()
            return await self.transcribe_video(video_path)
        else:
            return await super().run(input_text) 