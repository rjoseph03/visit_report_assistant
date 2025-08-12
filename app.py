import streamlit as st
import asyncio
import nest_asyncio
import base64
import numpy as np
import io
import wave
import librosa
import soundfile as sf
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from assistant_class import VoiceAssistant  # your existing class

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
    try:
        import soundfile as sf

        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        buf_out = io.BytesIO()
        sf.write(buf_out, audio_data, 16000, format="WAV", subtype="PCM_16")
        return base64.b64encode(buf_out.getvalue()).decode("utf-8")
    except Exception as e:
        st.error(f"Audio conversion error: {str(e)}")
        return None


def main():
    st.title("Azure OpenAI Voice Assistant (Mic Recorder)")

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

    # Display chat history
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
        st.info("Click Record, speak, then Stop to send your message")

        audio_data = mic_recorder(
            start_prompt="🎙️ Record",
            stop_prompt="⏹️ Stop",
            just_once=True,
            format="wav",
            key="recorder",
        )

        # Save to session state so we don't lose it on rerun
        if audio_data and "bytes" in audio_data:
            st.session_state.last_audio = audio_data

        if "last_audio" in st.session_state:
            audio_data = st.session_state.last_audio
            base64_audio = convert_audio_to_base64(audio_data["bytes"])
            if base64_audio:
                st.session_state.state.messages.append(
                    {"role": "user", "content": "🎙️ Voice message"}
                )
                with st.chat_message("user"):
                    st.write("🎙️ Voice message")
                    st.audio(audio_data["bytes"], format="audio/wav")

                with st.spinner("Processing voice message..."):
                    try:
                        response = st.session_state.state.loop.run_until_complete(
                            st.session_state.state.assistant.interact(
                                "voice", base64_audio
                            )
                        )
                        st.session_state.state.messages.append(
                            {"role": "assistant", "content": response}
                        )
                        with st.chat_message("assistant"):
                            st.write(response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            # Clear after processing so it doesn't reprocess
            del st.session_state.last_audio
            st.rerun()


if __name__ == "__main__":
    main()
