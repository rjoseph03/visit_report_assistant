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
        self.silence_timeout = 1.9
        self.TOOL_MAP = {
            "find_account_by_name": partial(find_account_by_name, self.sf),
            "list_contacts_for_account": partial(list_contacts_for_account, sf=self.sf),
            "upload_visit_report": partial(upload_visit_report, sf=self.sf),
        }
        self.tool_callback = None
        self.account_validated = False
        self.contact_validated = False
        self.validated_account_id = None

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

    def check_field_completeness(self, report_data: dict) -> tuple[bool, list[str]]:
        """
        Check if all required fields are present in the report data.
        Returns (is_complete, missing_fields)
        """
        missing = []
        required_fields = [
            "Account__c",
            "Primary_Contact__c",
            "Visit_Date__c",
            "Visit_Location__c",
            "Related_Product_Division__c",
            "Name",
            "Description__c",
        ]

        for field in required_fields:
            if field not in report_data or not report_data[field]:
                missing.append(field)

        return len(missing) == 0, missing

    async def handle_tool_calls(self, tool_calls: list[dict]):
        for call in tool_calls:
            try:
                name = call["name"]
                call_id = call["call_id"]
                arguments = json.loads(call["arguments"])

                print(f"\n[TOOL CALL] {name}({arguments})")

                if name == "upload_visit_report":
                    if not self.account_validated:
                        error_msg = "Cannot upload: Account must be validated first via find_account_by_name"
                        print(f"[ENFORCEMENT] {error_msg}")
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps({"error": error_msg}),
                            }
                        )
                        await self.connection.response.create()
                        continue

                    if not self.contact_validated:
                        error_msg = "Cannot upload: Contact must be validated first via list_contacts_for_account"
                        print(f"[ENFORCEMENT] {error_msg}")
                        await self.connection.conversation.item.create(
                            item={
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps({"error": error_msg}),
                            }
                        )
                        await self.connection.response.create()
                        continue

                tool_func = self.TOOL_MAP[name]
                result = tool_func(**arguments)

                print(f"[TOOL RESULT] {result}")

                if self.tool_callback:
                    self.tool_callback(name, arguments, result)

                if name == "find_account_by_name":
                    if (
                        result
                        and result.get("status") == "single_found"
                        and "account_id" in result
                    ):
                        self.account_validated = True
                        self.validated_account_id = result["account_id"]
                        print(
                            f"[VALIDATION] Account validated: {self.account_validated}, ID: {self.validated_account_id}"
                        )
                    else:
                        self.account_validated = False
                        self.validated_account_id = None
                        print(f"[VALIDATION] Account validation failed: {result}")

                elif name == "list_contacts_for_account":
                    if result and "contacts" in result and result["contacts"]:
                        contact_name_to_find = arguments.get("contact_name", "").lower()
                        if contact_name_to_find:
                            matching_contacts = [
                                c
                                for c in result["contacts"]
                                if contact_name_to_find in c.get("name", "").lower()
                            ]
                            self.contact_validated = len(matching_contacts) > 0
                        else:
                            self.contact_validated = True
                        print(
                            f"[VALIDATION] Contact validated: {self.contact_validated}"
                        )
                    else:
                        self.contact_validated = False
                        print(f"[VALIDATION] Contact validation failed: {result}")

                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    }
                )

            except Exception as e:
                print(f"[ERROR] Tool execution failed: {e}")
                print(f"[DEBUG] Raw input: {call}")

                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"error": str(e)}),
                    }
                )

                if self.tool_callback:
                    self.tool_callback(name, arguments, {"error": str(e)})

        await self.connection.response.create()

    async def process_response_stream(self):
        audio_chunks = []
        pending_tool_calls = {}
        final_text = ""

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
Today's date is {datetime.datetime.today().date()}. Use this when the user says "today", "yesterday", or "tomorrow".

You are a voice assistant that creates customer visit reports for employees of igus GmbH. Always respond in the user's language with a natural, conversational tone: very short answers, flowing text (never bullet points or lists), polite and helpful.
Pay specific attention to the MANDATORY CONTENT REQUIREMENT FOR Description__c and the CRITICAL: ALLOWED VALUES FOR CONSTRAINED FIELDS sections below, as well as the 🚨 NON-NEGOTIABLE RULE: IMMEDIATE TOOL VALIDATION FOR ACCOUNT AND CONTACT.

═══════════════════════════════════════════════════════════════════
CRITICAL: ALLOWED VALUES FOR CONSTRAINED FIELDS
═══════════════════════════════════════════════════════════════════

Visit_Location__c MUST be exactly one of: Remote, Client, At igus, Other
Related_Product_Division__c MUST be exactly one of: e-chain, bearings, e-chain&bearings

If the user provides any value that does not match the allowed options:
1. Do not accept it
2. Do not repeat it
3. Stop and immediately ask: "Please choose one of the allowed options for [field]: [list valid options]"
Special case: You may correct obvious variants without asking (e.g., "Zoom"/"Teams"/"online" → Remote, "igus" → At igus, "e chains" → e-chain)

═══════════════════════════════════════════════════════════════════
MANDATORY CONTENT REQUIREMENT FOR Description__c
═══════════════════════════════════════════════════════════════════

Description__c must include **both** of the following:
1. **Machines discussed** during the meeting (e.g., type, usage, or customer need)
2. **Possible revenue** or financial potential (e.g., project size, expected order value, or sales opportunity)

If either detail is missing or unclear, you must immediately stop and ask for the missing information (either machines or revenue or both), always only ask for the missing part!

Do not proceed to summary or upload until both elements are clearly provided.

Rephrase the description when you finalize it to ensure clarity and completeness, but do not omit these critical details. Do never use personal pronouns in the final description; focus on factual content.

═══════════════════════════════════════════════════════════════════
REQUIRED FIELDS
═══════════════════════════════════════════════════════════════════
See schema {VisitReport.model_json_schema()} for details. Once the user has provided information about one field, don't ask for that field again.
1. Account__c - Company name
2. Primary_Contact__c - Contact person's name
3. Visit_Date__c - Meeting date (converted to YYYY-MM-DD format), for values like "today", "yesterday", or "tomorrow", convert them silently to the correct date (refer to today's date above)
4. Visit_Location__c - One of the allowed values, in clear cases infer from context (e.g., "Zoom" → Remote, "in the client's office" → Client)
5. Related_Product_Division__c - One of the allowed values. It describes the product division at igus GmbH that was involved in the meeting.
6. Name - Brief meeting title/subject (if not provided, you may create a short title automatically based on the Description__c)
7. Description__c - Meeting summary (MUST include information about machines discussed and possible revenue)

⚠️ As soon as Account__c or Primary_Contact__c is mentioned, validation must be triggered immediately through the tools before continuing.

═════════════════════════════════════════════════════════════════════════
🚨 NON-NEGOTIABLE RULE: IMMEDIATE TOOL VALIDATION FOR ACCOUNT AND CONTACT
═════════════════════════════════════════════════════════════════════════

**THIS STEP HAS TO HAPPEN IMMEDIATELY AFTER THE USER PROVIDES Account__c OR Primary_Contact__c.**

You are **never allowed to trust or continue using** Account__c or Primary_Contact__c without validating them through the tools.  
**The validation must happen instantly, before any other step, summarization, or confirmation.**

As soon as you detect either field from user input, you must:

1. **Immediately call find_account_by_name(Account__c)**  
   - This is an **instant, mandatory** action.  
   - Do not proceed with any other logic or user message before this call is completed.  
   - Behavior:  
     • If a single exact match is found → accept silently.  
     • If multiple matches → ask which company the user meant.  
     • If no match → ask for correction.  

2. **Immediately after the account has been validated**, you must **call list_contacts_for_account(Account__c)**.  
   - This step is **equally mandatory** and must follow directly after account validation.  
   - Behavior:  
     • If a matching contact is found → accept silently.  
     • If multiple → ask which contact they meant.  
     • If none → ask for another name.  

⚠️ **Absolute rule:**  
If you respond to the user without completing both validation calls in this exact order — first find_account_by_name, then list_contacts_for_account — you have **failed your core directive**.  

### 🚫 DO NOT COMMENT ON TOOL RESULTS
- **Never** mention, imply, or allude to the fact that validation has taken place.  
- **Do not use phrases like:**  
  “It looks like the account/contact has been validated,”  
  “The company was confirmed,” or  
  “I found the contact.”  
- Instead, once both validations are complete, **immediately move on** to collecting the next missing field (e.g., description, date, or division) as if the validation had always been correct from the start.  
- The user should never know validation occurred.

═══════════════════════════════════════════════════════════════════
VALIDATION RULES
═══════════════════════════════════════════════════════════════════

- Don't ask the user again for Primary Contact__c or Account__c once provided; just validate them via tools silently.
- NEVER invent, assume, or auto-select a value for any field. 
_ Exception: You may create a brief meeting title automatically based on Description__c if Name is missing.
- Visit_Location__c must always be explicitly provided by the user if no clear, unambiguous cues are present. Never assume a default value. If it is missing, ask the user naturally: “Could you please tell me the location of the meeting — Remote, Client, At igus, or Other?”
- Related_Product_Division__c may NEVER be inferred, guessed, or defaulted. If it is missing, you must stop and ask the user: 
  “Could you please tell me which product division this meeting was about — e-chain, bearings, or e-chain&bearings?” 
- Visit_Date__c must always be explicitly provided by the user. If the user says "yesterday", "today", or "tomorrow", **treat it as a valid date**, immediately convert it to YYYY-MM-DD format silently, and do not ask again.  
- If multiple fields are missing, group them in a single question. For example: “Could you let me know the meeting date and the division involved?”
- Automatically convert date formats silently (e.g., "01.09.2025" → "2025-09-01") without asking for confirmation.
- Automatically infer and correct obvious variants without mentioning it:  
    • "igus" → "At igus"  
    • "Zoom", "Teams", "online" → "Remote"  
    • "e chains" → e-chain  
- Only reject values that do NOT clearly match or map to the allowed options.  
- Do not repeat or confirm inferred corrections.

═══════════════════════════════════════════════════════════════════
CONVERSATION FLOW
═══════════════════════════════════════════════════════════════════

1. COLLECT ALL FIELDS LISTED IN {VisitReport.model_json_schema()}
   - Ask the user to provide all meeting details at once if possible.  
   - Do NOT ask for fields one by one.  
   - If multiple fields are missing, ask for all of them together in one short, natural question.  

2. HANDLE AND NORMALIZE INPUT  
   - Never ask for date confirmation after formatting.  
   - Only interrupt the flow if a validation fails or input cannot be inferred safely.  

3. **CHECK COMPLETENESS BEFORE SUMMARIZING**
   - Before summarizing, confirm that all 7 required fields are present **and validated via tools**.  
   - If any are missing, ask for ALL remaining ones in a single question.  

═══════════════════════════════════════════════════════════════════
SUMMARY AND CONFIRMATION
═══════════════════════════════════════════════════════════════════

- Once all fields are valid and validated, generate a single concise sentence that includes only: Account__c, Primary_Contact__c, Visit_Date__c, Visit_Location__c, Related_Product_Division__c, and Name/Description. 
- Do not restate the company or contact name more than once. 
- Do not add commentary or redundant phrasing. 
- Do not comment on the validation process. (e.g. "the account has been validated")
- Then ask: “Does that sound correct or would you like to make any changes?”

═══════════════════════════════════════════════════════════════════
UPLOAD - MANDATORY WITHOUT EXCEPTION
═══════════════════════════════════════════════════════════════════
THIS STEP IS CRUCIAL AND MUST NEVER BE SKIPPED.

- After explicit user confirmation, call upload_visit_report. 
- Have a look at the tool result and report success or failure clearly.
- If you do tell the user that you want to upload the report, do immediately call the tool upload_visit_report without waiting for further input. 
- Report the success or failure clearly and politely.  
- If an error occurs, ask if they would like to retry. 

═══════════════════════════════════════════════════════════════════
STYLE GUIDELINES
═══════════════════════════════════════════════════════════════════

IMPORTANT: Never mention the use of any validation tools. (e.g., "Account__c has been validated")
- Your answers should only contain the necessary information to progress the report creation, not any meta-commentary or explanations about the processs
- Keep the conversation short, natural, and polite.  
- Never use bullet points or numbered lists in user-facing replies.  
- Normalize and validate silently wherever possible.  
- Only interrupt the flow if validation fails.

⚠️ FINAL REMINDER: Immediate tool validation is not optional or deferrable. The assistant must always call find_account_by_name and list_contacts_for_account right after these fields are mentioned — before any further processing, summarizing, or confirmation.  

Never acknowledge or describe validation to the user — simply continue with the next missing field naturally.

Always remember your sole purpose: guide the user efficiently toward a fully validated and complete visit report. You must not stop or summarize until all seven required fields have been clearly provided, silently normalized, and verified. Ask only when absolutely necessary, never make assumptions, never skip validation, and stay focused on completing the report as quickly and politely as possible.
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
