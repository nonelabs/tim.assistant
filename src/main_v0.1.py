# tim.assistant v0.1
# Copyright (c) 2024 Thomas Fritz, gematik GmbH
# Email: thomas.fritz@gematik.de
# License: MIT License

import os
import asyncio
import argparse
from datetime import datetime
from typing import Union
from llama_cpp import Llama
from nio import AsyncClient, MatrixRoom, RoomMessageText, InviteMemberEvent
from nio import RoomMessageAudio
from dotenv import load_dotenv
load_dotenv()

llm = Llama(
    model_path="./models/gemma-2-27b-it-Q5_K_M.gguf",
    n_gpu_layers=-1,  
    seed=1337,  
    n_ctx=4092,  
    verbose=True
)

anamnese = {}
USERNAME = os.environ["TIM_USERNAME"]
PASSWORD = os.environ["TIM_PASSWORD"]
HOMESERVER = os.environ["TIM_HOMESERVER"]

START_TIME = int(datetime.now().timestamp()*1000)
LAST_MSG = int(datetime.now().timestamp() * 1000)
messages = [{"role": "user",
             "content": "Du bist ein deutschsprachiger Assistent. Fordere mich auf mit der Anamnese zu beginnen und stell mir nacheinander die folgenden Fragen zu meinem "
                        "Gesundheitszustand. Stelle immer nur eine Frage! \n "
                        "1. Jetzige Anamnese: Was sind Ihre aktuellen Beschwerden? oder Seit wann treten die Symptome auf? \n"

                        "2. Frühere Anamnese: Hatten Sie in der Vergangenheit ähnliche Symptome? oder Haben Sie früher schon einmal dieselbe Erkrankung gehabt? \n"

                        "3. Medikamentenanamnese: Nehmen Sie derzeit Medikamente ein? oder „Haben Sie Allergien oder Unverträglichkeiten gegenüber bestimmten Medikamenten?\n "

                        "4. Vegetative Anamnese: Haben Sie Schlafstörungen? oder Haben Sie Veränderungen in Ihrem Appetit oder Gewicht bemerkt? \n" 

                        "5. Familienanamnese: Gibt es in Ihrer Familie Vorerkrankungen wie Diabetes oder Herzerkrankungen? oder Hatten Ihre Eltern oder Geschwister ähnliche Gesundheitsprobleme wie Sie? \n "

                        "6. Soziale Anamnese: Haben Sie einen Beruf oder eine Tätigkeit, die relevant für Ihre Gesundheit sein könnte? oder Sind Sie verheiratet, ledig, geschieden oder verwitwet? \n"

                        "7. Psychiatrische Anamnese: Haben Sie in der Vergangenheit jemals eine psychische Erkrankung gehabt? oder Haben Sie derzeit Stress oder Probleme in Ihrem persönlichen oder beruflichen Leben? \n"

                        "8. Eigenanamnese: Wie würden Sie Ihre aktuelle körperliche und mentale Gesundheit beschreiben? oder Haben Sie in letzter Zeit Veränderungen in Ihrem Gesundheitszustand bemerkt? \n" 

                        "Wenn du alle Informationen hast, bedanke dich und fasse sie in einem FHIR bundle als JSON zusammen. Verwende dabei FHIR Observation für Beschwerden und FHIR Conditions für Erkrankungen. Füge jeweils eine Zusammenfassung der relevanten Informationen als Note hinzu .Verwende kein Markdown nur Plain Text. Dann verabschiede dich. \n"},
            {"role": "model", "content": "Verstanden!"}
            ]


async def message_callback(room: MatrixRoom, event: Union[RoomMessageText, RoomMessageAudio]) -> None:
    global LAST_MSG
    global messages
    if event.server_timestamp > LAST_MSG and event.sender != f"@{USERNAME}:{HOMESERVER}":
        LAST_MSG = 1e100
        if isinstance(event, RoomMessageText) and len(event.body) > 0:
            message_content = event.body
        else:
            return
        messages.append({"role": "user",
                         "content": "Anweisung: Leite den Patienten bei den Antworten. Wenn der Patient Fragen hat oder sich unsicher ist oder die Frage nicht ausreichend beantwortet, "
                                    "dann gehe darauf ein und hilf ihm deine Fragen vollständig zu beantworten. Die Antwort sollte sehr detailiert sein und dem Arzt helfen. Stelle die "
                                    "nächste Frage erst, wenn die aktuelle vollständig beantwortet wurde. Stelle immer nur eine Frage nach der anderen !!!" + message_content})
        x = llm.create_chat_completion(messages=messages)
        output = x["choices"][0]["message"]["content"]
        await client.room_send(
            room_id=room.room_id,
            message_type="m.room.message",
            content={"msgtype": "m.text", "body": output},
        )
        messages.append({"role": "model", "content": output})
        LAST_MSG = int(datetime.now().timestamp() * 1000)


async def invite_callback(room: MatrixRoom, event: InviteMemberEvent) -> None:
    print(f"Received invite to {room.room_id} from {event.sender}")
    await client.join(room.room_id)
    await client.room_send(
        room_id=room.room_id,
        message_type="m.room.message",
        content={"msgtype": "m.text", "body": "Thanks for the invite!"},
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='tim.assistant v0.2')
    parser.add_argument('-u', '--username', required=False,
                        help='username')
    parser.add_argument('-p', '--password', required=False,
                        help='password')
    parser.add_argument('-s', '--homeserver', required=False,
                        help=f'TIM assistant homeserver')
    return parser.parse_args()

async def main() -> None:
    global client

    client = AsyncClient(f"https://{HOMESERVER}", USERNAME)
    await client.login(PASSWORD)
    client.add_event_callback(message_callback, (RoomMessageText, RoomMessageAudio))
    client.add_event_callback(invite_callback, InviteMemberEvent)
    await client.sync_forever(timeout=30000)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
