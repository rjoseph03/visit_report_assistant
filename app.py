import streamlit as st
import asyncio
import nest_asyncio
from assistant_class import VoiceAssistant
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()


class SessionState:
    def __init__(self):
        self.assistant = None
        self.messages = []
        self.tool_calls = []
        self.loop = asyncio.new_event_loop()


@st.cache_resource
def get_assistant():
    loop = asyncio.new_event_loop()

    async def init_assistant():
        assistant = VoiceAssistant()
        await assistant.connect()
        return assistant

    return loop.run_until_complete(init_assistant()), loop


def tool_callback(tool_name, arguments, result):
    if "state" in st.session_state:
        st.session_state.state.tool_calls.append(
            {"name": tool_name, "arguments": arguments, "result": result}
        )


def display_tool_calls():
    if st.session_state.state.tool_calls:
        st.sidebar.header("Tool Calls")
        for i, tool_call in enumerate(st.session_state.state.tool_calls):
            with st.sidebar.expander(
                f"Tool Call {i+1}: {tool_call['name']}", expanded=False
            ):
                st.code(f"Arguments: {tool_call['arguments']}", language="json")
                st.code(f"Result: {tool_call['result']}", language="json")


def main():
    st.title("Voice Assistant for creating Visit Reports")

    if "state" not in st.session_state:
        st.session_state.state = SessionState()
        try:
            assistant, loop = get_assistant()
            st.session_state.state.assistant = assistant
            st.session_state.state.loop = loop
            st.session_state.state.assistant.tool_callback = tool_callback
        except Exception as e:
            st.error(f"Failed to initialize assistant: {str(e)}")
            st.stop()

    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Input Mode", ("Text", "Voice"))
        if st.button("Clear Conversation"):
            st.session_state.state.messages = []
            st.session_state.state.tool_calls = []
            st.rerun()

    display_tool_calls()

    for message in st.session_state.state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if mode == "Text":
        if prompt := st.chat_input("Type your message"):
            st.session_state.state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Assistant is thinking..."):
                try:
                    response = st.session_state.state.loop.run_until_complete(
                        st.session_state.state.assistant.interact("text", prompt)
                    )
                    st.session_state.state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    with st.chat_message("assistant"):
                        st.write(response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
            st.rerun()

    elif mode == "Voice":
        st.info(
            "Click Record and speak, then the assistant will process your message automatically."
        )
        if st.button("🎙️ Record Voice Message"):
            with st.spinner("Recording and processing..."):
                try:
                    response = st.session_state.state.loop.run_until_complete(
                        st.session_state.state.assistant.interact("voice")
                    )
                    st.session_state.state.messages.append(
                        {"role": "user", "content": "🎙️ Voice message"}
                    )
                    with st.chat_message("user"):
                        st.write("🎙️ Voice message")
                    st.session_state.state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    with st.chat_message("assistant"):
                        st.write(response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            st.rerun()


if __name__ == "__main__":
    main()
