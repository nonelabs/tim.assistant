# tim.assistant v1.0
# Copyright (c) 2024 Thomas Fritz, gematik GmbH
# Email: thomas.fritz@gematik.de
# License: MIT License

import os
import argparse
import threading
import json
import requests
import torch
import aiofiles
from pydub import AudioSegment
from nio import RoomMessageAudio, DownloadError
import whisper
from typing import Union
import tempfile
import asyncio
import time
from nio import AsyncClient, MatrixRoom, RoomMessageText, InviteMemberEvent
from datetime import datetime
from medical_history import MedicalHistory
from dotenv import load_dotenv

whisper_model = None

load_dotenv()
USERNAME = os.environ["TIM_USERNAME"]
PASSWORD = os.environ["TIM_PASSWORD"]
HOMESERVER = os.environ["TIM_HOMESERVER"]
QUERY_GROUPS_FOLDER = "./query_groups/"
START_TIME = int(datetime.now().timestamp() * 1000)

medical_history_data = {}
queue = {}
bundles = {}


def upload_bundle(bundle, FHIR_SERVER, auth_token=None):
    headers = {"Content-Type": "application/fhir+json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return requests.post(f"{FHIR_SERVER}/Bundle", json=bundle, headers=headers).json()


def check_medical_history():
    while True:
        user_completed = None
        for k, v in medical_history_data.items():
            if v.status == "completed":
                queue[k]["lock"].acquire()
                bundle = v.generate_fhir_bundle()
                bundles[k] = bundle
                with open(k + ".json", "w") as file:
                    json.dump(bundle, file, indent=True)
                user_completed = k
                break
        if user_completed:
            del medical_history_data[user_completed]
            queue[user_completed]["lock"].release()
        time.sleep(10)


async def message_callback(
    room: MatrixRoom, event: Union[RoomMessageText, RoomMessageAudio]
) -> None:
    global queue
    global medical_history_data
    if (
        event.server_timestamp > START_TIME
        and (event.sender).lower() != f"@{USERNAME}:{HOMESERVER}".lower()
    ):
        if isinstance(event, RoomMessageText) and len(event.body) > 0:
            message_content = event.body
        elif isinstance(event, RoomMessageAudio):
            if whisper_model:
                message_content = await audio_to_text(event, client)
            else:
                await client.room_send(
                    room_id=room.room_id,
                    message_type="m.room.message",
                    content={"msgtype": "m.text", "body": "No whisper model available"},
                )
                return
        else:
            raise Exception("Something went wrong")

        if event.sender in medical_history_data:
            medical_history = medical_history_data[event.sender]
        else:
            medical_history = MedicalHistory(
                event.sender, query_groups_folder=QUERY_GROUPS_FOLDER
            )
            queue[event.sender] = {"messages": "", "lock": threading.Lock()}
            medical_history_data[event.sender] = medical_history
        if not queue[event.sender]["lock"].locked():
            queue[event.sender]["lock"].acquire()
            if "fhir bundle" in message_content and event.sender in bundles:
                message = json.dumps(bundles[event.sender], indent=True)
            elif medical_history.status == "completed":
                message = (
                    "Die Ananmese ist bereits abgeschlossen und wird nun verarbeitet."
                )
            else:

                queued_messages = queue[event.sender]["messages"]
                message = (
                    message_content
                    if len(queued_messages) == 0
                    else queued_messages + "\n" + message_content
                )
                message = medical_history.next(message)
                queue[event.sender]["messages"] = str()
            await client.room_send(
                room_id=room.room_id,
                message_type="m.room.message",
                content={"msgtype": "m.text", "body": message},
            )
            queue[event.sender]["lock"].release()
        else:
            queue[event.sender]["messages"] += "\n" + message_content


async def audio_to_text(event: RoomMessageAudio, client: AsyncClient) -> str:
    try:
        mxc_url = event.url
        response = await client.download(mxc_url)
        if isinstance(response, DownloadError):
            return f"Error: Unable to download audio file. {response.message}"
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_filename = os.path.join(tmpdirname, f"temp_audio_{event.event_id}")
            async with aiofiles.open(temp_filename, "wb") as f:
                await f.write(response.body)
            try:
                audio = AudioSegment.from_file(temp_filename)
            except Exception as e:
                return f"Error: Unable to process audio file. {str(e)}"

            wav_filename = os.path.join(tmpdirname, f"temp_audio_{event.event_id}.wav")
            audio.export(wav_filename, format="wav")
            result = whisper_model.transcribe(wav_filename, language="de")
            text = result["text"]

        return text

    except Exception as e:
        return f"Error processing audio: {str(e)}"


async def invite_callback(room: MatrixRoom, event: InviteMemberEvent) -> None:
    await client.join(room.room_id)
    await client.room_send(
        room_id=room.room_id,
        message_type="m.room.message",
        content={"msgtype": "m.text", "body": "Thanks for the invite!"},
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tim.assistant v0.2")
    parser.add_argument("-u", "--username", required=False, help="username")
    parser.add_argument("-p", "--password", required=False, help="password")
    parser.add_argument(
        "-w", "--whisper", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-q", "--query_groups", required=False, help="query groups folder"
    )
    parser.add_argument(
        "-s", "--homeserver", required=False, help=f"TIM assistant homeserver"
    )
    return parser.parse_args()


async def main() -> None:
    global client
    global whisper_model
    global HOMESERVER
    global USERNAME
    global PASSWORD
    global QUERY_GROUPS_FOLDER

    args = parse_arguments()
    HOMESERVER = args.homeserver if args.homeserver else HOMESERVER
    USERNAME = args.username if args.username else USERNAME
    PASSWORD = args.password if args.password else PASSWORD
    QUERY_GROUPS_FOLDER = args.query_groups if args.query_groups else PASSWORD

    client = AsyncClient(f"https://{HOMESERVER}", USERNAME)
    monitor_thread = threading.Thread(target=check_medical_history)
    monitor_thread.daemon = True
    monitor_thread.start()
    if args.whisper:
        whisper_model = whisper.load_model("medium")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            whisper_model = whisper_model.to(device)
    await client.login(PASSWORD)
    client.add_event_callback(message_callback, (RoomMessageText, RoomMessageAudio))
    client.add_event_callback(invite_callback, InviteMemberEvent)
    print("tim.assistant running ...")
    await client.sync_forever(timeout=30000)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
