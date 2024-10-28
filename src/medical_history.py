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
    model_path="models/gemma-2-27b-it-Q3_K_L.gguf",
    n_gpu_layers=-1,
    seed=1337,
    n_ctx=4092,
    verbose=False,
)
model_lock = threading.Lock()

class MedicalHistory:
    def __init__(self, subject_reference, query_group_folder):
        self.subject_reference = subject_reference
        self.query_groups = []
        self.current_query_group = None
        self.query_results = []
        self.query_group_iter = None
        self.context = []
        self.questions_for_medical_staff = []
        self.additional_questions = []
        self.fhir_export_finished = False
        self.status = "pending"
        self.load_query_groups(query_group_folder)

    def load_query_groups(self):
        queries_folder = "query_groups_small/"
        for json_file in os.listdir(queries_folder):
            if json_file.endswith("-observations.json"):
                file_path = os.path.join(queries_folder, json_file)
                self.query_groups.append(load_query_group_from_json(file_path))
        for json_file in os.listdir(queries_folder):
            if json_file.endswith("-conditions.json"):
                file_path = os.path.join(queries_folder, json_file)
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
            "The last message of the user was: "
            + last_message
            + " Is it an answer to a question, a question or an instruction ?",
            ["answer", "question", "instruction"],
        )
        if res == "answer":
            return self.process_answer()
        if res == "question":
            return self.process_question()
        if res == "instruction":
            instruction = self.context[-2]['content']
            del self.context[-2:]
            return self.internal_instruction(
                "Antworten sie auf die Nachricht des Patienten: "
                + instruction
                + ". Erklären Sie freundlich, dass er sich erstmal auf medizinische Befragung fokussieren soll " \
                  "und Sie sich gerne nachher noch Zeit für andere Fragen oder Anregungen nehmen werden"
            )

    def internal_instruction(self, task, options=None, true_value=None):
        context = self.context.copy()
        if context[-1]["role"] == "user":
            context.pop()
        if options is not None:
            grammar = (
                    "root ::= choices\nchoices ::= ("
                    + re.sub("[,']", "", str(['"' + v + '"|' for v in options]))[1:-2]
                    + ")"
            )
            context.append(
                {
                    "role": "user",
                    "content": "Instruction: "
                               + task
                               + "Select your answer from:"
                               + grammar[1:-1].replace("|", ", " + "!"),
                }
            )
            grammar = LlamaGrammar.from_string(grammar)
        else:
            grammar = None
            context.append({"role": "user", "content": "Instruction: " + task})
        with model_lock:
            x = model.create_chat_completion(
                grammar=grammar, messages=context, temperature=0
            )
        res = x["choices"][0]["message"]["content"]
        print("========================= internal instruction =====================")
        print(self.context[-2])
        print(self.context[-1])
        print("Task: " + task),
        print("Options: " + str(options)),
        print("Result:" + res)
        if true_value is not None:
            print("Boolean" + str(res == true_value))
        print("=====================================================================")
        return res if true_value is None else res == true_value

    def is_medical_staff_required(self):
        if self.internal_instruction(
                'Your task is to determine if the message of the user is related to the medical assessment. Answer with "yes" or "no"!',
                ["yes", "no"],
                "yes",
        ):
            if self.internal_instruction(
                    'You are tasked with determining if the message or question of user requires professional medical '
                    'expertise to be answered safely and accurately. Answer with "yes" if the question should only be '
                    'answered by qualified healthcare professionals, or "no" if it can be discussed by non-medical persons.',
                    ["yes", "no"],
                    "yes",
            ):
                summary = self.internal_instruction(
                    "Summarize the question of the user in German!"
                )
                self.questions_for_medical_staff.append(summary)
                del self.context[-1:]
                self.context.append(
                    {
                        "role": "model",
                        "content": self.internal_instruction(
                            'Erklären Sie dem Patienten, dass diese Frage von einer medizinischen Fachkraft beantwortet werden muss und sie'
                            'daher weitergeleitet wird und man auf ihn zukommen wird. Sag ihm, dass folgende Frage weitergeleitet wurde:'
                            + summary
                        ),
                    }
                )
                return True
            else:
                return False
        else:
            return False

    def process_question(self):
        options = [v.name for _, v in self.current_query_group.queries.items()]
        options.append("Sonstiges")
        question = self.context[-2]["content"]
        res = self.internal_instruction(
            "Your are tasked with classifying the best matching context of the last question of user based on the dialogue? Answer in German with just one word! ",
            options,
        )
        if res == "Sonstiges":
            if not self.is_medical_staff_required():
                question = self.context[-2]['content']
                del self.context[-2:]
                return self.internal_instruction(
                    'Antworten Sie kurz und knapp auf die Frage des Patienten: '
                    + question
                    + '. Markieren Sie die Antwort mit (NICHT VERIFIZIERT) Falls die Frage nicht im Zusammenhang mit der Befragung steht, so '
                      'erklären Sie freundlich, dass er sich erstmal auf die medizinische Befragung fokussieren soll.' 
                      'und Sie sich gerne nachher noch Zeit für andere Fragen oder Anregungen nehmen werden'
                )
        else:
            for i, (k, v) in enumerate(self.current_query_group.queries.items()):
                if res == v.name:
                    if self.internal_instruction(
                            "You have the following Information: "
                            + v.additional_info
                            + "The patient has the following question: "
                            + question
                            + ' Determine if you can answer the question of the patient based on the given information.! Answer with "yes" or "no"',
                            ["yes", "no"],
                    ):
                        del self.context[-1:]
                        self.context.append(
                            {
                                "role": "model",
                                "content": self.internal_instruction(
                                    "Information: "
                                    + v.additional_info
                                    + ". Beantworte folgende Frage basierend auf der Information: "
                                    + question
                                    + ". Markieren Sie die Antwort mit (VERIFIZIERT). Dann fahre mit der Befragung fort. Nutze kein Markdown!"
                                ),
                            }
                        )
                        return None
                    else:
                        self.is_medical_staff_required()
                        return None
        return None

    def process_answer(self):
        if self.internal_instruction(
                'Determine from the answer of the assistant if the assessment of the medical history has been finished! '
                'Answer with "yes" if it has been finished or "no" if not!',
                ["yes", "no"],
                "yes",
        ):
            self.current_query_group.summary = self.internal_instruction(
                "Fasse alle genannten Beschwerden und genannten Vorerkrankungen des Patienten zusammen."
            )
            self.current_query_group = next(self.query_group_iter, "end")
            if not self.current_query_group == "end":
                self.context = [
                    {
                        "role": "user",
                        "content": 'Sie sind ein medizinischer Assistent. Sie halten sich immer kurz und verwenden niemals Markdown. Sie sind gerade '
                                   'im Gespräch mit einem Patienten und führen eine Anamnese mit ihm durch. Stellen Sie direkt die nächsten Fragen '
                                   'ohne sich vorzustellen oder Begrüßung!'
                                   + self.get_query_group_question_context(
                            self.current_query_group
                        ),
                    }
                ]
                with model_lock:
                    x = model.create_chat_completion(messages=self.context, temperature=0)
                self.context.append(
                    {"role": "model", "content": x["choices"][0]["message"]["content"]}
                )
            else:
                summary = "Patientenbericht: "
                questions = "Fragen an das medizinische Personal:"
                for g in self.query_groups:
                    summary += g.summary + "\n"
                if len(self.questions_for_medical_staff) > 0:
                    for q in self.questions_for_medical_staff:
                        questions += q + "\n"
                else:
                    questions += "keine."
                self.context = [
                    {
                        "role": "user",
                        "content": 'Sie sind ein medizinischer Assistent, Sie halten sich immer kurz und verwenden niemals Markdown. Sie haben '
                                   'gerade ein Anamnese Gespräch abgeschlossen! Folgendes hat sich ergeben:'
                                   + summary
                                   + 'Sagen Sie dem Patienten, dass die Befragung fertig ist. Fassen Sie den Patientenbericht kurz zusammen. '
                                     'Folgende Fragen sind gehen an das medizinische Personal: '
                                   + questions
                                   + ' Fassen Sie diese auch kurz zusammen Fassen Sie diese auch kurz zusammen und verabschieden Sie sich.',
                    }
                ]
                with model_lock:
                    x = model.create_chat_completion(messages=self.context, temperature=0)
                self.status = "completed"
                return x["choices"][0]["message"]["content"]
        else:
            if not self.internal_instruction(
                    'Are the last messages of user and model still in context of the medical history assessment addressing '
                    'the issues annotated mit "PUNKTE:"',
                    ["yes", "no"],
                    "yes",
            ):
                del self.context[-1:]
                self.context.append(
                    {
                        "role": "model",
                        "content": self.internal_instruction(
                            "Nenne dem Patienten dazu auch nochmal alle noch offenen Punkte !"
                        ),
                    }
                )
                return None

    def generate_fhir_bundle(self):
        for g in self.query_groups:
            if len(g.summary) > 0:
                for _, q in g.queries.items():
                    print(q.name)
                    grammar = 'root ::= choices\nchoices ::= ("true"|"false")'
                    grammar = LlamaGrammar.from_string(grammar)
                    context = [
                        {
                            "role": "user",
                            "content": "Patientenbericht: "
                                       + g.summary
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
        print(json.dumps(bundle, indent=True))
        return bundle

    def next(self, user_message):
        if self.status == "pending":
            self.query_group_iter = iter(self.query_groups)
            self.current_query_group = next(self.query_group_iter)
            self.context = [
                {
                    "role": "user",
                    "content": 'Sie sind ein medizinischer Assistent names Anna, Sie halten sich immer kurz verwenden niemals Markdown. '
                               'Stellen Sie sich kurz vor. Erklären Sie dem Patienten, dass Sie eine Reihe von Fragen zu seinem aktuellen '
                               'Gesundheitszustand und seiner Krankengeschichte stellen um sich ein umfassendes Bild zu machen.'
                               + self.get_query_group_question_context(self.current_query_group),
                }
            ]
            self.status = "inprogress"
        elif self.status is "completed":
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
