# tim.assistant v1.0
# Copyright (c) 2024 Thomas Fritz, gematik GmbH
# Email: thomas.fritz@gematik.de
# License: MIT License

import json
import os
import re
import threading
from datetime import datetime

from llama_cpp import Llama, LlamaGrammar

from query import QueryResult, QueryGroup, load_query_group_from_json

model = Llama(
    model_path="models/gemma-2-27b-it-Q5_K_M.gguf",
    n_gpu_layers=-1,
    seed=1337,
    n_ctx=4092,
    verbose=False,
)
model_lock = threading.Lock()

class MedicalHistory:
    def __init__(self, subject_reference, query_groups_folder):
        self.subject_reference = subject_reference
        self.query_groups = []
        self.current_query_group = None
        self.query_results = []
        self.query_group_iter = None
        self.context = []
        self.questions_for_medical_staff = ""
        self.additional_questions = []
        self.fhir_export_finished = False
        self.status = "pending"
        self.misc = []
        self.report = "Patientenbericht:\n"
        self.load_query_groups(query_groups_folder)

    def load_query_groups(self, query_groups_folder):
        for json_file in os.listdir(query_groups_folder):
            if json_file.endswith("-observations.json"):
                file_path = os.path.join(query_groups_folder, json_file)
                self.query_groups.append(load_query_group_from_json(file_path))
        for json_file in os.listdir(query_groups_folder):
            if json_file.endswith("-conditions.json"):
                file_path = os.path.join(query_groups_folder, json_file)
                self.query_groups.append(load_query_group_from_json(file_path))

    def add_query_result(self, query_result):
        self.query_results.append(query_result)

    def get_query_group_question_context(self, query_group: QueryGroup):
        context = query_group.prompt + "PUNKTE: "
        for i, (k, v) in enumerate(query_group.queries.items()):
            context += v.name + ","
        context = context[:-1]
        return context

    def review_messages(self):
        if len(self.context) < 3:
            return None
        last_message = self.context[-2]["content"]
        res = self.internal_instruction(
            'Die letzte Nachricht des Nutzers lautet:"'
            + last_message
            + '" Klassifiziere: Ist die Nachricht eine Antwort auf eine gestellte Frage (Antwort), '
                                'ist die Nachricht eine Frage (Frage) oder ist die Nachricht eine Anweisung (Anweisung)?',
            ["Antwort", "Frage", "Anweisung"]
        )
        if res == "Antwort":
            return self.process_answer()
        if res == "Frage":
            return self.process_question()
        if res == "Anweisung":
            instruction = self.context[-2]['content']
            del self.context[-2:]
            return self.internal_instruction(
                "Antworten sie auf die Nachricht des Patienten: \""
                + instruction
                + "\". Erklären Sie freundlich, dass er sich erstmal auf medizinische Befragung fokussieren soll " \
                  "und Sie sich gerne nachher noch Zeit für andere Fragen oder Anregungen nehmen werden. Fahre dann mit der Befragung fort. Falls es hilfreich erscheint, nenne nochmal die offnenen Punkte."
            )

    def internal_instruction(self, task, options=None, true_value=None, use_context=True):
        if use_context:
            context = self.context.copy()
        else:
            context = [] 
        if options is not None:
            grammar = (
                    "root ::= choices\nchoices ::= ("
                    + re.sub("[,']", "", str(['"' + v + '"|' for v in options]))[1:-2]
                    + ")"
            )
            context.append(
                {
                    "role": "user",
                    "content": "Anweisung: "
                               + task
                               + "Wähle deine Antwort aus den folgenden Möglichkeiten:"
                               + grammar[1:-1].replace("|", ", ") + "!"
                }
            )
            grammar = LlamaGrammar.from_string(grammar)
        else:
            grammar = None
            context.append({"role": "user", "content": "Anweisung: " + task})
        with model_lock:
            x = model.create_chat_completion(
                grammar=grammar, messages=context, temperature=0
            )
        res = x["choices"][0]["message"]["content"]
        print("==================== Internal instruction =====================")
        print("User:  "+self.context[-2]['content'])
        print("Model: "+self.context[-1]['content'])
        print("Task: " + task),
        if options is not None:
            print("Options: " + str(options)),
        print("Result:" + res)
        return res if true_value is None else res == true_value

    def medical_staff_required(self):
        question = self.context[-2]['content']
        if self.internal_instruction(
            "Die Nachricht des Nutzers lautet:\""
            + question
            + '" Klassifiziere: Handelt es sich bei der Nachricht um eine Frage, die medizinisch relevant ist und im Kontext der Befragung steht ?'
            ' Antworte mit "ja" oder "nein".',
            ["ja", "nein"],"ja"
        ):
            if self.internal_instruction(
                'Die Nachricht des Nutzers lautet: "'
                + question
                + '" Klassifiziere: Handelt es sich bei der Nachricht um eine Frage, auf die ausschließlich eine medizinische '
                'Fachkraft eingehen sollte, dann antworten sie mit "ja". Falls es sich um eine unkritische Frage handelt, die im Kontext der Befragung von einem KI System beantwortet werden kann, antworten Sie mit "nein".',
                ["ja", "nein"],"ja"
            ):
                summary = self.internal_instruction(
                    "Fasse die Frage des Nutzers im Kontext der Befragung knapp und neutral zusammen. Verwende kein Markdown !"
                )
                self.questions_for_medical_staff += "Frage:" + summary + "\n"
                del self.context[-2:]
                return self.internal_instruction(
                            'Erklären Sie dem Patienten, dass diese Frage von einer medizinischen Fachkraft beantwortet werden muss und sie'
                            'daher weitergeleitet wird. Sag ihm, dass die folgende Nachricht an die Medizinische Fachkraft geschickt wird. Nachricht: "'
                            + summary + '". Versichere dem Patienten, dass die medizinische Fachkraft auf ihn zukommen wird. Verwende kein Markdown!'
                        )
            else:
                return None 
        else:
            return None
    def classify_message(self):
        options = [v.name for _, v in self.current_query_group.queries.items()]
        options.append("Sonstiges")
        message = self.context[-2]["content"]
        res = self.internal_instruction(
            "Die Frage Nachricht des Nutzers lautet:\""
            + message
            + "\" Klassifiziere: Zu welchen dieser Themen passt die Frage am Besten ?",
            options
        )
        return res

    def process_question(self):
        res = self.classify_message()
        question = self.context[-2]['content']
        if res == "Sonstiges":
            ms = self.medical_staff_required()
            if ms is None:
                del self.context[-2:]
                return self.internal_instruction(
                    'Antworten Sie kurz und knapp auf die Frage des Patienten: '
                    + question
                    + '. Falls die Frage nicht im Zusammenhang mit den aktuellen Punkten steht, so '
                      'erklären Sie freundlich, dass sich der Patient erstmal auf die medizinische Befragung zu den aktuellen Punkten fokussieren soll' 
                      ' und Sie sich gerne nachher noch Zeit für andere Fragen oder Anregungen nehmen werden. '
                      'Wiederhole dann nochmal die letzte Frage an den Patienten.'
                )
            else:
                return ms
        else:
            for i, (k, v) in enumerate(self.current_query_group.queries.items()):
                if res == v.name:
                    if self.internal_instruction(
                            "Sie haben die folgenden verifizierten Informationen: "
                            + v.additional_info
                            + "Der Patient hat die folgende Frage gestellt: "
                            + question
                            + ' Können Sie die Frage basierend auf den verifizierten Informationen beantworten ? Antworten Sie mit "ja" or "nein"',
                            ["ja", "nein"],"ja",use_context=False
                    ):
                        self.context[-1] = {
                                "role": "model",
                                "content": self.internal_instruction(
                                    "Information: "
                                    + v.additional_info
                                    + ". Beantworte folgende Frage basierend auf der Information: "
                                    + question
                                    + ". Danach stelle entweder Fragen im Kontext der Frage des Patienten oder nenne dem Patienten nochmal alle offenen Punkte!"
                                ),
                            }
                        return None
                    else:
                        return self.medical_staff_required()
        return None

    def process_answer(self):
        if self.internal_instruction(
                'Wurde die medizinische Befragung des Patienten zu seinem Gesundheitszustand beendet ?'
                'Antworte mit "ja" oder "nein"!',
                ["ja", "nein"],
                "ja",
        ):
            self.report += self.internal_instruction(
                "Erstelle eine vollständige Zusammenfassung aller Beschwerden und Vorerkrankungen von denen der Patient berichtet hat!"
            )

            self.current_query_group = next(self.query_group_iter, None)
            if not self.current_query_group is None: 
                self.context = [
                    {
                        "role": "user",
                        "content": 'Sie sind ein medizinischer Assistent. Sie halten sich immer kurz und verwenden niemals Markdown. Sie sind gerade '
                                   'im Gespräch mit einem Patienten und führen eine Anamnese mit ihm durch. Stellen Sie direkt die nächsten Fragen '
                                   'ohne sich vorzustellen oder Begrüßung! Befragen sie den Patienten nur zu den im folgenden genannten Punkten. Falls '
                                   'der Patient andere Erkrankungen oder Beschwerden anspricht, dann lenke das Gespräch auf die Punkte'
                                   + self.get_query_group_question_context(self.current_query_group) 
                                  + ' Beenden Sie danach das Gespräch'
                    }
                ]
                with model_lock:
                    x = model.create_chat_completion(messages=self.context, temperature=0)
                self.context.append(
                    {"role": "model", "content": x["choices"][0]["message"]["content"]}
                )
            else:
                questions = "Fragen an das medizinische Personal:\n"
                if len(self.questions_for_medical_staff) > 0:
                        questions += self.questions_for_medical_staff
                else:
                    questions += " keine."
                self.context = [
                    {
                        "role": "user",
                        "content": 'Sie sind ein medizinischer Assistent, Sie sprechen immer Deutsch und halten sich immer kurz und verwenden niemals Markdown. Sie haben '
                                   'gerade ein Anamnese Gespräch abgeschlossen! Folgendes hat sich ergeben:'
                                   + self.report 
                                   + 'Sagen Sie dem Patienten, dass die Befragung fertig ist. Fassen Sie den Patientenbericht kurz zusammen. '
                                     'Folgende Fragen sind gehen an das medizinische Personal: '
                                   + questions
                                   + ' Fassen Sie diese auch kurz zusammen und verabschieden Sie sich.',
                    }
                ]
                with model_lock:
                    x = model.create_chat_completion(messages=self.context, temperature=0)
                self.status = "completed"
                return x["choices"][0]["message"]["content"]
        else:
            answer = self.context[-2]['content']
            if self.internal_instruction(
                "Die Nachricht des Nutzers lautet:\""
                + answer
                + '" Klassifiziere: Handelt es sich bei der Nachricht um eine Aussage oder Frage, die medizinisch relevant ist und im Kontext der Befragung steht ?'
                ' Antworte mit "ja" oder "nein".',
                ["ja", "nein"],"ja"
            ):
                return None
            else:
                del self.context[-2:]
                return self.internal_instruction(
                    'Gehen Sie kurz und knapp auf die Aussage des Patienten ein: '
                    + answer
                    + '. Sagen Sie ihm dann er solle sich erstmal auf die Befragung fokussieren, bzw. plausible Antworten geben.'
                    'Wiederhole dann nochmal die letzte Frage an den Patienten.'
                )
            return None

    def generate_fhir_bundle(self):
        for g in self.query_groups:
            for _, q in g.queries.items():
                print(q.name)
                grammar = 'root ::= choices\nchoices ::= ("true"|"false")'
                grammar = LlamaGrammar.from_string(grammar)
                context = [
                    {
                        "role": "user",
                        "content": "Patientenbericht: "
                                    + self.report
                                    + " Anweisung:"
                                    + q.match_prompt,
                    }
                ]
                with model_lock:
                    x = model.create_chat_completion(
                        grammar=grammar, messages=context, temperature=0
                    )
                res = x["choices"][0]["message"]["content"]
                if res == "true":
                    if q.type == "Observation":
                        context.append(
                            {
                                "role": "model",
                                "content": "Der Patient berichtet ueber folgende Beschwerde:"
                                            + str(q.name),
                            }
                        )
                        context.append(
                            {
                                "role": "user",
                                "content": "Extrahiere aus dem Patientenbericht die relevanten Informationen zu "
                                            + str(q.name)
                                            + " und fasse sie zusammen",
                            }
                        )
                        with model_lock:
                            x = model.create_chat_completion(
                                messages=context, temperature=0
                            )
                        res = x["choices"][0]["message"]["content"]
                        self.add_query_result(
                            QueryResult(self.subject_reference, q, res)
                        )
                    elif q.type == "Condition":
                        context.append(
                            {
                                "role": "model",
                                "content": 'Der Patient berichtet ueber folgende chronische oder bestehende '
                                            'diagnostizierte Erkrankungen:'
                                            + str(q.name),
                            }
                        )
                        context.append(
                            {
                                "role": "user",
                                "content": 'Extrahiere aus dem Patientenbericht die relevanten Informationen zu '
                                            + str(q.name)
                                            + " und fasse sie zusammen",
                            }
                        )
                        with model_lock:
                            x = model.create_chat_completion(
                                messages=context, temperature=0
                            )
                        res = x["choices"][0]["message"]["content"]
                        self.add_query_result(
                            QueryResult(self.subject_reference, q, res)
                        )

        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "timestamp": datetime.now().isoformat(),
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": self.subject_reference,
                        "active": True,
                    }
                }
            ],
        }

        for q in self.query_results:
            resource = q.to_fhir()
            entry = {
                "fullUrl": f"urn:uuid:{resource['id']}",
                "resource": resource.copy(),
            }
            bundle["entry"].append(entry)
        return bundle

    def next(self, user_message):
        if self.status == "pending":
            self.query_group_iter = iter(self.query_groups)
            self.current_query_group = next(self.query_group_iter)
            self.context = [
                {
                    "role": "user",
                    "content": 'Sie sind ein medizinischer Assistent, Sie sprechen immer Deutsch. Sie halten sich immer kurz verwenden und niemals Markdown. '
                               'Stellen Sie sich kurz vor. Erklären Sie dem Patienten, dass Sie eine Reihe von Fragen zu seinem aktuellen '
                               'Gesundheitszustand und seiner Krankengeschichte stellen um sich ein umfassendes Bild zu machen. '
                               'Befragen sie den Patienten nur zu den im folgenden genannten Punkten. Falls er andere Erkrankungen oder Beschwerden anspricht, '
                               'dann lenke das Gespräch auf die folgenden Punkte'
                               + self.get_query_group_question_context(self.current_query_group) + 'Beenden Sie danach das Gespräch'
                }
            ]
            self.status = "inprogress"
        elif self.status == "completed":
            return "Die Anamnese wurde bereits erhoben"
        else:
            self.context.append({"role": "user", "content": user_message})
        with model_lock:
            x = model.create_chat_completion(messages=self.context, temperature=0)
        self.context.append(
            {"role": "model", "content": x["choices"][0]["message"]["content"]}
        )
        reviewed_answer = self.review_messages()
        return (
            self.context[-1]["content"] if reviewed_answer is None else reviewed_answer
        )
