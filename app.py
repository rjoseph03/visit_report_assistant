import streamlit as st
import asyncio
import nest_asyncio
import base64
import numpy as np
import io
import wave
from dotenv import load_dotenv
from assistant_class import VoiceAssistant

load_dotenv()
nest_asyncio.apply()


class SessionState:
    def __init__(self):
        self.assistant = None
        self.messages = []
        self.loop = asyncio.new_event_loop()


@st.cache_resource
def get_assistant():
    loop = asyncio.new_event_loop()

    async def init_assistant():
        assistant = VoiceAssistant()
        await assistant.connect()
        return assistant

    return loop.run_until_complete(init_assistant()), loop


def convert_audio_to_base64(audio_bytes):
    if audio_bytes is None:
        return None

    try:
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())

        wav_data = buf.getvalue()
        return base64.b64encode(wav_data).decode("utf-8")
    except Exception as e:
        st.error(f"Audio conversion error: {str(e)}")
        return None


def main():
    st.title("Voice Assistant Chat")

    if "state" not in st.session_state:
        st.session_state.state = SessionState()
        try:
            assistant, loop = get_assistant()
            st.session_state.state.assistant = assistant
            st.session_state.state.loop = loop
        except Exception as e:
            st.error(f"Failed to initialize assistant: {str(e)}")
            st.stop()

    with st.sidebar:
        st.header("Controls")
        mode = st.radio("Input Mode", ("Text", "Voice"))
        if st.button("Clear Conversation"):
            st.session_state.state.messages = []
            st.rerun()

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
        st.info("Click the microphone button below to record your message")

        audio_bytes = st.audio_input("Record your voice message")

        if audio_bytes is not None:
            st.audio(audio_bytes, format="audio/wav")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "🎙️ Send Voice Message", type="primary", use_container_width=True
                ):
                    base64_audio = convert_audio_to_base64(audio_bytes)

                    if base64_audio:
                        st.session_state.state.messages.append(
                            {"role": "user", "content": "🎙️ Voice message"}
                        )

                        with st.chat_message("user"):
                            st.write("🎙️ Voice message")
                            st.audio(audio_bytes, format="audio/wav")

                        with st.spinner("Processing voice message..."):
                            try:
                                response = (
                                    st.session_state.state.loop.run_until_complete(
                                        st.session_state.state.assistant.interact(
                                            "voice", base64_audio
                                        )
                                    )
                                )
                                st.session_state.state.messages.append(
                                    {"role": "assistant", "content": response}
                                )

                                with st.chat_message("assistant"):
                                    st.write(response)

                                st.success("Voice message processed!")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                        st.rerun()
                    else:
                        st.error("Failed to process audio")

            with col2:
                if st.button("🗑️ Clear Recording", use_container_width=True):
                    st.rerun()


if __name__ == "__main__":
    main()
