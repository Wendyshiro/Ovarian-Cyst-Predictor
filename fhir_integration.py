# Refactored FHIR Integration Module for Ovarian Cyst Prediction System

import json
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Optional, Any
import uuid

class FHIRIntegration:
    def __init__(self, fhir_base_url: str = "http://hapi.fhir.org/baseR4"):
        self.fhir_base_url = fhir_base_url
        self.headers = {
            'Content-Type': 'application/fhir+json',
            'Accept': 'application/fhir+json'
        }

    def create_patient_resource(self, patient_data: Dict) -> Dict:
        patient_id = patient_data.get('patient_id', str(uuid.uuid4()))
        
        return {
            "resourceType": "Patient",
            "id": patient_id,
            "identifier": [
                {
                    "system": "http://hospital.example.com/patients",
                    "value": patient_id
                }
            ],
            "active": True,
            "name": [
                {
                    "use": "official",
                    "text": f"Patient {patient_id}"
                }
            ],
            "gender": patient_data.get('gender', 'female'),
            "birthDate": self._calculate_birth_date(patient_data.get('age', 0)),
            "address": [
                {
                    "use": "home",
                    "text": patient_data.get('region', 'Unknown'),
                    "city": patient_data.get('region', 'Unknown'),
                    "country": "KE"
                }
            ]
        }

    def create_observation_resource(self, patient_data: Dict, observation_type: str) -> Dict:
        observation_id = str(uuid.uuid4())

        loinc_codes = {
            'cyst_size': '38269-7',
            'ca125': '10334-1',
            'age': '30525-0',
            'ultrasound': '59776-5'
        }

        code = loinc_codes.get(observation_type, '38269-7')
        subject_ref = {"reference": f"Patient/{patient_data.get('patient_id', 'unknown')}"}

        observation = {
            "resourceType": "Observation",
            "id": observation_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": code,
                        "display": observation_type.replace('_', ' ').title()
                    }
                ]
            },
            "subject": subject_ref,
            "effectiveDateTime": datetime.now().isoformat(),
        }

        if observation_type == 'ultrasound':
            observation["valueBoolean"] = True
        else:
            observation["valueQuantity"] = {
                "value": self._get_observation_value(patient_data, observation_type),
                "unit": self._get_observation_unit(observation_type),
                "system": "http://unitsofmeasure.org",
                "code": self._get_observation_unit_code(observation_type)
            }

        return observation

    def create_condition_resource(self, patient_data: Dict, prediction_result: Dict) -> Dict:
        condition_id = str(uuid.uuid4())

        return {
            "resourceType": "Condition",
            "id": condition_id,
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active"
                    }
                ]
            },
            "verificationStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed",
                        "display": "Confirmed"
                    }
                ]
            },
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                            "code": "problem-list-item",
                            "display": "Problem List Item"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": "1234567890",
                        "display": "Ovarian Cyst"
                    }
                ],
                "text": "Ovarian Cyst"
            },
            "subject": {
                "reference": f"Patient/{patient_data.get('patient_id', 'unknown')}"
            },
            "onsetDateTime": datetime.now().isoformat(),
            "severity": {
                "coding": [
                    {
                        "system": "http://snomed.info/sct",
                        "code": self._get_severity_code(prediction_result),
                        "display": prediction_result.get('prediction', 'Unknown')
                    }
                ]
            }
        }

    def create_care_plan_resource(self, patient_data: Dict, care_template: Dict) -> Dict:
        care_plan_id = str(uuid.uuid4())
        return {
            "resourceType": "CarePlan",
            "id": care_plan_id,
            "status": "active",
            "intent": "plan",
            "title": f"Ovarian Cyst Care Plan - {patient_data.get('patient_id', 'Unknown')}",
            "description": "AI-generated care plan for ovarian cyst management",
            "subject": {
                "reference": f"Patient/{patient_data.get('patient_id', 'unknown')}"
            },
            "period": {
                "start": datetime.now().isoformat(),
                "end": self._calculate_end_date(care_template.get('treatment_protocol', {}))
            },
            "author": {
                "reference": "Practitioner/ai-system",
                "display": "AI Prediction System"
            },
            "careTeam": [
                {
                    "reference": "CareTeam/ovarian-cyst-team"
                }
            ],
            "activity": self._create_care_activities(care_template)
        }

    def send_bundle_to_fhir_server(self, resources: List[Dict]) -> Dict:
        bundle = {
            "resourceType": "Bundle",
            "type": "transaction",
            "entry": [
                {
                    "resource": res,
                    "request": {
                        "method": "POST",
                        "url": res["resourceType"]
                    }
                } for res in resources
            ]
        }

        try:
            url = f"{self.fhir_base_url}/"
            response = requests.post(url, json=bundle, headers=self.headers)
            return {
                "success": response.status_code in [200, 201],
                "status_code": response.status_code,
                "response": response.json()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calculate_birth_date(self, age: int) -> str:
        birth_year = datetime.now().year - age
        return f"{birth_year}-01-01"

    def _get_observation_value(self, patient_data: Dict, observation_type: str) -> float:
        return {
            'cyst_size': patient_data.get('cyst_size', 0),
            'ca125': patient_data.get('ca125_level', 0),
            'age': patient_data.get('age', 0)
        }.get(observation_type, 0)

    def _get_observation_unit(self, observation_type: str) -> str:
        return {
            'cyst_size': 'cm',
            'ca125': 'U/mL',
            'age': 'years'
        }.get(observation_type, 'unknown')

    def _get_observation_unit_code(self, observation_type: str) -> str:
        return {
            'cyst_size': 'cm',
            'ca125': 'U/mL',
            'age': 'a'
        }.get(observation_type, 'unknown')

    def _get_severity_code(self, prediction_result: Dict) -> str:
        return {
            'Observation': '255604002',
            'Medication': '255605001',
            'Surgery': '255606000',
            'Referral': '255607009'
        }.get(prediction_result.get('prediction', 'Observation'), '255604002')

    def _calculate_end_date(self, treatment_protocol: Dict) -> str:
        months = int(treatment_protocol.get('duration', '1').split()[0])
        end_date = datetime.now() + relativedelta(months=+months)
        return end_date.isoformat()

    def _create_care_activities(self, care_template: Dict) -> List[Dict]:
        treatment_plan = care_template.get('ai_recommendation', {}).get('treatment_plan', 'Unknown')
        return [
            {
                "outcomeCodeableConcept": [
                    {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "1234567890",
                                "display": "Ovarian Cyst Management"
                            }
                        ]
                    }
                ],
                "detail": {
                    "kind": "ServiceRequest",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": "1234567890",
                                "display": treatment_plan
                            }
                        ]
                    },
                    "description": f"AI recommended treatment: {treatment_plan}"
                }
            }
        ]
