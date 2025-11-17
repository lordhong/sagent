from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from sagent.utils.nodes import call_model, should_continue, tool_node, chat_node, evaluator_node, bank_submission_node, email_node, image_ocr_node, audio_transcription_node, video_transcription_node, sms_node
from sagent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the two nodes we will cycle between
workflow.add_node("Orchestrator", call_model)
# workflow.add_node("action", tool_node)
workflow.add_node("Email Agent", email_node)
workflow.add_node("Image OCR Agent", image_ocr_node)
workflow.add_node("Audio Transcription Agent", audio_transcription_node)
workflow.add_node("Video Transcription Agent", video_transcription_node)
workflow.add_node("SMS Agent", sms_node)
workflow.add_node("Chat Agent", chat_node)
workflow.add_node("Evaluator", evaluator_node)
workflow.add_node("Bank Submission", bank_submission_node)



# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("Orchestrator")

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
# workflow.add_edge("Orchestrator", "action")
workflow.add_edge("Orchestrator", "Email Agent")
workflow.add_edge("Orchestrator", "Image OCR Agent")
workflow.add_edge("Orchestrator", "Audio Transcription Agent")
workflow.add_edge("Orchestrator", "Video Transcription Agent")
workflow.add_edge("Orchestrator", "SMS Agent")
workflow.add_edge("Orchestrator", "Chat Agent")

# workflow.add_edge("action", "Evaluator")
workflow.add_edge("Email Agent", "Evaluator")
workflow.add_edge("Image OCR Agent", "Evaluator")
workflow.add_edge("Audio Transcription Agent", "Evaluator")
workflow.add_edge("Video Transcription Agent", "Evaluator")
workflow.add_edge("SMS Agent", "Evaluator")
workflow.add_edge("Chat Agent", "Evaluator")



workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "Evaluator",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "Orchestrator",
        # Otherwise we finish.
        "end": "Bank Submission",
    },
)

workflow.add_edge("Bank Submission", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()