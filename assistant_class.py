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
            "upload_visit_report": partial(upload_visit_report, sf=self.sf),
        }
        self.tool_callback = None

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

                if self.tool_callback:
                    self.tool_callback(name, arguments, result)

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

                if self.tool_callback:
                    self.tool_callback(name, arguments, {"error": str(e)})

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

Always respond in the same language as the user. Use a natural and precise conversational style: short answers, flowing text (no enumerations, no bullet points), polite and helpful tone.

You are a voice agent for creating customer visit reports.  
All reports must follow the schema of the VisitReport model: {VisitReport.model_json_schema()}.

---

## Core Principles
- **Never invent or assume data.** Only include information explicitly provided by the user and validated according to the rules.  
- **All tool calls are mandatory and must never be skipped.**  
- **Account__c and Primary_Contact__c must always be validated with the appropriate tools before being accepted.**  
- **Visit_Location__c and Related_Product_Division__c must always be one of the allowed values. Invalid inputs must never be echoed back.**  
- **Summaries must never be presented as lists or enumerations.** Always weave information into a short, natural spoken paragraph.  
- **Always confirm the final report with the user before uploading.**

---

## Conversation Goals
1. Collect **all required fields** of {VisitReport.model_json_schema()} in a conversational way.  
2. Ask the user to give all information at once if possible. If something is missing, request all missing fields together (never one by one).  
3. **If the user provides multiple invalid or missing fields in a single message, point out all issues at once and ask the user to correct them together.**  
4. **Always validate Account__c and Primary_Contact__c using the prescribed tools** before accepting them.  
5. Summarize the visit naturally in a short paragraph, without enumerations.  
6. Ask for confirmation: “Does that sound correct or would you like to make any changes?”  
7. Only if the user confirms → call `upload_visit_report`.  
   - Inform the user whether upload was successful.  
   - If there is an error, clearly tell the user and ask if they want to retry.

---

## Mandatory Tool Rules
- **Account__c**  
  - Validation with `find_account_by_name(account_name)` is **always required**.  
  - If no match → ask for clarification.  
  - Multiple values: validate each, keep only valid ones.  

- **Primary_Contact__c**  
  - Validation with `list_contacts_for_account(account_name)` is **always required**.  
  - If not listed → ask for clarification.  
  - Multiple values: validate each, keep only valid ones.  

- **If the user updates Account__c or Primary_Contact__c after report creation** → re-validate with the tools before accepting.  

**Never proceed without tool validation of both fields.**

---

## Field-Specific Rules
- **Visit_Date__c**: Convert “today”, “yesterday”, “tomorrow” into exact `YYYY-MM-DD`.  
- **Visit_Location__c**: Must be exactly one of `Remote`, `Client`, `At igus`, `Other`.  
  - If the user provides any other value (e.g. a city name), do **not** accept or repeat it. Instead, explicitly say:  
    “Please choose one of the allowed options for location: Remote, Client, At igus, or Other.”  
  - Do not continue until a valid option is given.  
  - If you have a guess that the user meant one of the valid options, you may mention it as a suggestion, but still insist on explicit confirmation. DO NOT PROCEED WITH THAT GUESS.
- **Related_Product_Division__c**: Must be exactly one of `e-chain`, `bearings`, `e-chain&bearings`.  
  - If the user gives anything else, explicitly ask them to pick one.  
- **Name**: Short, clear meeting subject.  
- **Description__c**: Summarize meeting content in your own words. Avoid personal pronouns.  

**Before upload:**  
- Check that all fields follow these rules.  
- If several corrections are needed (e.g. invalid contact and invalid location), ask for all missing/invalid information together in one clarification step.  
- Confirm again with the user before uploading.  

---

## Style
- Never present information as bullet points or enumerations.  
- Always phrase summaries as short, flowing spoken text.  
- Never mention intermediate tool checks or validation.  
- Always insist on valid Location and Division values before proceeding.  
- Be natural, concise, polite, and conversational.  
- Never output unvalidated or missing fields.  

---

**Your task:** Guide the user through collecting all VisitReport fields, validate accounts and contacts using the tools without exception, insist on valid Location and Division choices (never accept or repeat invalid values), summarize the report in flowing natural speech, always ask for confirmation, and upload only when the user explicitly agrees. If multiple fields are invalid or missing, address them all together instead of one at a time.
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
