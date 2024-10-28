# tim.assistant v1.0
# Copyright (c) 2024 Thomas Fritz, gematik GmbH
# Email: thomas.fritz@gematik.de
# License: MIT License

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import json

@dataclass
class Query:
    name: str
    type: str
    code: Dict[str, Any]
    inquiry_prompt: str
    match_prompt: str
    additional_info: str


class QueryResult:
    def __init__(self, subject_reference, query: Query, note: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.subject_reference = subject_reference
        self.query = query
        self.date_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.note = note

    def to_fhir(self):
        fhir_resource = {
            "id": self.id,
            "code": self.query.code,
            "subject": {
                "reference": self.subject_reference,
            },
            "note": [{"text": self.note}],
        }
        if self.query.type == "condition":
            fhir_resource["resourceType"] = "Condition"
            fhir_resource["meta"]: {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Condition"]
            }
            fhir_resource["onsetDateTime"] = self.date_time
        elif self.query.type == "Observation":
            fhir_resource["resourceType"] = "Observation"
            fhir_resource["meta"]: {
                "profile": ["http://hl7.org/fhir/StructureDefinition/Observation"]
            }
            fhir_resource["effectiveDateTime"] = self.date_time
        return fhir_resource


class QueryGroup:
    def __init__(
        self,
    ):
        self.queries = {}
        self.prompt = str()
        self.summary = str()


def load_query_group_from_json(file_path: str):
    query_group = QueryGroup()
    with open(file_path, "r") as file:
        data = json.load(file)
    query_group.prompt = data["prompt"]
    for key, value in data["query_group"].items():
        query_group.queries[key] = Query(**value)
    return query_group