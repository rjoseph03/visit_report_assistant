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

from simple_salesforce import Salesforce
from openai import AsyncAzureOpenAI
from tools import (
    find_account_by_name,
    list_contacts_for_account,
    prepare_for_upload,
    upload_visit_report,
    TOOLS,
)
from models import VisitReport
from functools import partial


class VoiceAssistant:
    def __init__(self, model="gpt-4o-mini-realtime-preview", sample_rate=16000):
        self.model = model
        self.sample_rate = sample_rate
        self.client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2025-04-01-preview",
        )
        self.sf = Salesforce(
            username=os.getenv("SF_USER"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_TOKEN"),
            domain="test",
        )
        self.connection = None
        self.vad = webrtcvad.Vad(2)
        self.frame_duration = 30
        self.silence_timeout = 1.5
        self.TOOL_MAP = {
            "find_account_by_name": partial(find_account_by_name, self.sf),
            "list_contacts_for_account": partial(list_contacts_for_account, sf=self.sf),
            "prepare_for_upload": prepare_for_upload,
            "upload_visit_report": partial(upload_visit_report, sf=self.sf),
        }

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

                tool_func = self.TOOL_MAP[name]
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
                "instructions": f"""
Today's date is {datetime.datetime.today().date()}.

Do always respond with the same language as the user.

You are a voice agent for creating customer visit reports in the format of the VisitReport model: {VisitReport.model_json_schema()}.

---

## Goal
- Gather all required VisitReport fields from the user in a conversational way, always with help of the tools
- Do not request them one by one; instead, ask for all missing information together.
- Validate **AccountName** and **PrimaryContact** using the correct tools before accepting them into the report.
- Make the information about the visit ready for Upload by using the tool `prepare_for_upload`.

---

## MANDATORY TOOL CALL RULES

***You must ALWAYS use the following tools exactly as described.***  
***Do not record AccountName or PrimaryContact without first validating through the tools. ***  
Never guess — only trust tool output and user confirmation.

1. **If the user gives an AccountName** →  
   - Call `find_account_by_name(account_name)` to validate.  
   - If the tool finds no match → ask the user for clarification.

2. **If the user gives a PrimaryContact** →  
   - Call `list_contacts_for_account(account_name)` to check if the contact is listed for that account.  
   - If not listed → ask for clarification.

3. **If the user has given an AccountName but NOT a PrimaryContact** →  
   - Call `list_contacts_for_account(account_name)` to get the available contacts.  
   - Ask the user to choose one from the list.

4. **If the user provides multiple AccountName or PrimaryContact options** →  
   - Validate each one with the relevant tool call(s).  
   - Include only the valid ones in the field, separated by commas.  
   - If any are invalid → ask the user for clarification.

5. **If the user updates AccountName or PrimaryContact after the report is created** →  
   - Repeat the relevant validation tool calls before accepting the new value.

---

## Processing Rules
- **Date**: Convert “today”, “yesterday”, “tomorrow” into the exact `YYYY-MM-DD` date.  
  Date field must always be in `YYYY-MM-DD` format.

- **Location** and **Division**: Accept any variation, but record exactly what was said.

- **Description**: Summarize what the user described in your own words, avoid using personal pronouns.

---

## Output Rules
1. When all fields are collected:  
   - Create a short, friendly spoken summary weaving all facts into 2–3 sentences, do not explicitly confirm tool lookup results, unless you need to ask for clarification.  
   - Do not list fields — make it sound like everyday conversation.  
   - Keep all details accurate.

2. After summary:  
   - Ask: “Does that sound correct or would you like to make any changes?”

3. If the user confirms:  
   - Call `prepare_for_upload`.  
   - Tell the user: “Your visit report is now ready for upload to Salesforce.”  
   - Ask if they would like to create another report.

---

## Style
- Always speak responses aloud.  
- Stay conversational, polite, and helpful.  
- Never skip a mandatory tool call.  
- Never assume or invent data.
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
