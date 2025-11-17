# SAGENT

**S**ystem for **A**utonomous **G**raph-Enhanced Multimodal LLM **A**gents

A multi-agent orchestration system built with LangGraph that routes tasks to specialized agents for multimodal processing and communication.

## Overview

SAGENT uses a graph-based workflow to coordinate specialized agents:
- **Orchestrator**: Routes tasks to appropriate agents
- **Evaluator**: Determines task completion and workflow continuation
- **Bank Submission**: Handles final task submission

## Specialized Agents

- **Email Agent**: Gmail integration for email management
- **Chat Agent**: Intercom integration for customer conversations
- **SMS Agent**: Twilio integration for SMS messaging
- **Image OCR Agent**: Text extraction from images (Google Gemini)
- **Audio Transcription Agent**: Speech-to-text conversion (GPT-4)
- **Video Transcription Agent**: Video content transcription (GPT-4)
- **Bank Submission Agent**: Stripe integration for dispute submissions

## Architecture

Built on LangGraph with a state-based workflow:
```
Orchestrator → Specialized Agents → Evaluator → [Continue | Bank Submission] → END
```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

- `OPENAI_API_KEY`: For OpenAI models
- `GOOGLE_API_KEY`: For Gemini vision models
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`: For SMS
- `INTERCOM_API_KEY`, `INTERCOM_ADMIN_ID`: For chat
- `STRIPE_API_KEY`: For bank submissions
- Gmail OAuth credentials: `credentials.json` and `token.pickle`

## Usage

```python
from sagent.agent import graph

# Run the workflow
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Your task here"}]},
    config={"configurable": {"model_name": "openai"}}
)
```

## License

See [LICENSE](LICENSE) file.
