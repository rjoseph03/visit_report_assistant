import os
import base64
import asyncio
import sounddevice as sd
import numpy as np
import io
import wave
import json
import webrtcvad
import datetime

from openai import AsyncAzureOpenAI
from tools import (
    find_account_by_name,
    list_contacts_for_account,
    prepare_for_upload,
    TOOLS,
    TOOL_MAP,
)
from models import VisitReport


class VoiceAssistant:
    def __init__(self, model="gpt-4o-mini-realtime-preview", sample_rate=16000):
        self.model = model
        self.sample_rate = sample_rate
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-04-01-preview",
        )
        self.connection = None
        self.vad = webrtcvad.Vad(2)
        self.frame_duration = 30
        self.silence_timeout = 1.5

    def record_until_silence(self):
        frame_size = int(self.sample_rate * self.frame_duration / 1000)
        silence_limit = int(self.silence_timeout * 1000 / self.frame_duration)

        audio_frames = []
        silence_counter = 0

        print("Recording... Speak now.")

        with sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="int16", blocksize=frame_size
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(frame_size)
                audio_chunk = audio_chunk[:, 0]

                pcm_bytes = audio_chunk.tobytes()
                audio_frames.append(pcm_bytes)

                is_speech = self.vad.is_speech(pcm_bytes, self.sample_rate)

                if is_speech:
                    silence_counter = 0
                else:
                    silence_counter += 1
                    if silence_counter >= silence_limit:
                        print("Silence detected. Stopping.")
                        break

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(audio_frames))

        return base64.b64encode(buf.getvalue()).decode()

    def play_audio_buffered(self, base64_chunks: list, sample_rate=24000):
        if not base64_chunks:
            print("No audio to play.")
            return

        raw_bytes = b"".join(base64.b64decode(chunk) for chunk in base64_chunks)
        audio_array = np.frombuffer(raw_bytes, dtype=np.int16)
        head_padding = np.zeros(int(0.1 * sample_rate), dtype=np.int16)
        tail_padding = np.zeros(int(0.2 * sample_rate), dtype=np.int16)
        padded_audio = np.concatenate([head_padding, audio_array, tail_padding])

        sd.play(padded_audio, samplerate=sample_rate)
        sd.wait()

    async def handle_tool_calls(self, tool_calls: list[dict]):
        for call in tool_calls:
            try:
                name = call["name"]
                call_id = call["call_id"]
                arguments = json.loads(call["arguments"])

                print(f"\n[TOOL CALL] {name}({arguments})")

                tool_func = TOOL_MAP[name]
                result = tool_func(**arguments)

                print(f"[TOOL RESULT] {result}")

                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    }
                )

                await self.connection.response.create()

            except Exception as e:
                print(f"[ERROR] Tool execution failed: {e}")
                print(f"[DEBUG] Raw input: {call}")

    async def process_response_stream(self):
        audio_chunks = []
        pending_tool_calls = {}

        async for event in self.connection:
            if event.type == "response.audio.delta":
                audio_chunks.append(event.delta)

            elif event.type == "response.audio_transcript.done":
                final_text = event.transcript.strip()

            elif event.type == "response.function_call_arguments.delta":
                call_id = event.call_id
                if call_id not in pending_tool_calls:
                    pending_tool_calls[call_id] = {"arguments": [], "name": None}
                pending_tool_calls[call_id]["arguments"].extend(event.delta)

            elif event.type == "response.function_call_arguments.done":
                call_id = event.call_id
                if call_id in pending_tool_calls:
                    pending_tool_calls[call_id]["name"] = event.name

            elif event.type == "response.done":
                if pending_tool_calls:
                    tool_calls = []
                    for call_id, call_data in pending_tool_calls.items():
                        if call_data["name"]:
                            args_str = "".join(call_data["arguments"])
                            print(f"[TOOL ARGS RAW] {args_str}")
                            print(f"[TOOL NAME] {call_data['name']}")

                            tool_calls.append(
                                {
                                    "name": call_data["name"],
                                    "call_id": call_id,
                                    "arguments": args_str,
                                }
                            )

                    if tool_calls:
                        await self.handle_tool_calls(tool_calls)
                        return "tool_called"
                else:
                    if final_text:
                        print(f"\n[ASSISTANT RESPONSE] {final_text}")

                    if audio_chunks:
                        self.play_audio_buffered(audio_chunks)
                    else:
                        print("[DEBUG] No audio to play")

                    return final_text

    async def connect(self):
        self.connection = await self.client.beta.realtime.connect(
            model=self.model
        ).__aenter__()

        await self.connection.session.update(
            session={
                "modalities": ["text", "audio"],
                "tools": TOOLS,
                "tool_choice": "auto",
                "instructions": f"""Today's date is {datetime.datetime.today().date()}.

You are a voice agent for creating customer visit reports. You should listen to what the user says and collect information for a visit report. Once you have all information, respond with the summary or, if you are unsure, ask for clarification.

Make sure that you have information about all required fields and pay attention to the data types: {VisitReport.model_json_schema()}.

You shouldn't ask the user to report each information step by step, but ask for all information in one go.

**Goal:**  
Collect these 7 fields:  
- AccountName (company the meeting was with)  
- PrimaryContact  
- Date  
- Location (options: remote or client or igus or other)  
- Division (options: e-chain or bearings or e-chain&bearings)  
- Subject  
- Description  
- Optional: machines that were topic

**Process:**  
- Listen to the user and match the given information to the fields.  

- If the user gives you the AccountName, use the tool `find_account_by_name` to find the account and ask for clarification if you cannot find it (tool call is mandatory).  

- If the user gives you the PrimaryContact, use the tool `list_contacts_for_account` to look whether the contact is in the list for the account and ask for clarification if you cannot find it (tool call is mandatory).  

- If the user gave you an AccountName but not PrimaryContact, execute the tool `list_contacts_for_account` to get the list of contacts and ask the user to choose one (tool call is mandatory).  

- If the user wants to update AccountName or PrimaryContact after you created the report, do again use the tools to verify the information (tool call is mandatory).  

- If the user gives you several options for AccountName or PrimaryContact, verify both of them and include them if they are both valid, otherwise ask the user for clarification (tool call is mandatory). If there are several valid options, **ALWAYS** separate the with a comma (e.g. Max Msutermann, Peter Silie). 

- Use today's date when the user refers to the date of the meeting with expressions like "today", "yesterday", "tomorrow", but the field date always has to be in the style YYYY-MM-DD ***(the line with the date should lways look like this :**Date**: YYYY-MM-DD)*** 

- Do not include expressions like yesterday or tomorrow in the date, even when the user used it

- Information in division and location may differ from the main options, you do not necessarily need to use one of the main options, just pay close attention to their occurrence.  

- The field description should contain  a brief summary of the description the user gave you in your own words

- If you don't get any information on machines, insert None into that field

- All fields should just contain the information as the user gives it to you, no filling words like other etc.

- If you have gathered all required fields, create a brief structured summary in the style of {VisitReport.model_json_schema()}. Keep the order of paramters as I gave it to you in the beginning. (Do only respond with the structured summary and ask whether the user wants to make any changes).

- If the final summary was confirmed by the user, ALWAYS use the tool 'prepare_for_upload' and tell the user that the visit report is now ready for upload to Salesforce.

- Ask the user whether he wants to create another report.

**Rules:**  
- When you have all information, do always respond with an object like this: {VisitReport.model_json_schema()}.  
- Do not create additional fields to the VisitReport object.  
- Always use tools for lookup.  
- Only trust tool data and the information you are given by the user — never assume.  
- Speak responses aloud.  
- Stay conversational and helpful.

***After you gathered all information, make sure the report is in a valid {VisitReport.model_json_schema()} format and you follow all rules from the Process***

***You reply after you have gathereed all information, should directly be the structured response, not anything else.***
""",
            }
        )

    async def interact(self, mode: str, content: str = None):
        if mode == "voice":
            content = [{"type": "input_audio", "audio": self.record_until_silence()}]
        elif mode == "text":
            content = [{"type": "input_text", "text": content}]
        else:
            raise ValueError("Invalid mode. Choose 'text' or 'voice'.")

        await self.connection.conversation.item.create(
            item={"type": "message", "role": "user", "content": content}
        )
        await self.connection.response.create()

        while True:
            result = await self.process_response_stream()
            if result != "tool_called":
                return result
            await self.connection.response.create()


async def main():
    assistant = VoiceAssistant()
    await assistant.connect()

    while True:
        mode = (
            input("Type 't' for text, 'v' for voice, or 'q' to quit: ").strip().lower()
        )
        if mode == "q":
            print("Goodbye!")
            break

        if mode == "t":
            text = input("Enter a message: ")
            await assistant.interact("text", text)
        elif mode == "v":
            await assistant.interact("voice")
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    asyncio.run(main())
