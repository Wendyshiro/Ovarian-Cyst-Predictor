# FHIR Integration Module – Refactored Version

This module is part of the Ovarian Cyst Prediction System and is responsible for communicating with FHIR-compliant healthcare systems. It has been significantly refactored to improve **readability, maintainability, standard compliance**, and **bundle-level transaction support**.

## 🔧 Key Improvements Made

### 1. **Cleaner, Modular Design**
- Broke down logic into smaller, well-named helper functions (e.g., `_get_observation_value`, `_calculate_birth_date`).
- Introduced `relativedelta` for safe date arithmetic instead of manually changing months.
- Switched to `.get()` accessors throughout to avoid runtime crashes due to missing keys.

---

### 2. **Accurate Medical Standards**
- Updated LOINC and SNOMED CT codes:
  - `cyst_size`: `38269-7`
  - `ca125`: `10334-1`
  - `ultrasound`: `59776-5`
  - `age`: `30525-0`
- Used correct coding systems for condition severity.

---

### 3. **FHIR Transaction Bundles**
- Added a new method `send_bundle_to_fhir_server()` to transmit multiple resources in a single FHIR-compliant transaction bundle.
- This improves reliability, atomicity, and interoperability across systems.

---

### 4. **Ultrasound Observation Handling**
- Special-cased `ultrasound` as a `valueBoolean`, since it is binary (present or not).
- Other observations continue to use `valueQuantity`.

---

### 5. **Flexible Patient Demographics**
- Gender can now be passed dynamically (`female` as default).
- Patient address captures both `text` and `city` for regional traceability.

---

### 6. **Improved CarePlan Logic**
- Dynamically calculates care plan `end` date based on protocol duration.
- Supports activities tied to AI-generated recommendations.

---

### 7. **Production-Ready Practices**
- Structured the code to support future enhancements (e.g., authentication, retries, validation).
- Included extensibility for other standards like OpenHIE and DHIS2.

---

## 🚀 How to Use

### Initialize the FHIR Integration Class

```python
fhir = FHIRIntegration(fhir_base_url="http://hapi.fhir.org/baseR4")

#Create Resources
patient_resource = fhir.create_patient_resource(patient_data)
observation_resource = fhir.create_observation_resource(patient_data, "cyst_size")
condition_resource = fhir.create_condition_resource(patient_data, prediction_result)
care_plan_resource = fhir.create_care_plan_resource(patient_data, care_template)
#Send as a Bundle
fhir.send_bundle_to_fhir_server([
    patient_resource,
    observation_resource,
    condition_resource,
    care_plan_resource
])

# Sample Input Testing For Postman or curl
#Patient
{
  "patient_id": "OC-2001",
  "age": 45,
  "region": "Nairobi"
}
#Observation
{
  "patient_data": {
    "patient_id": "OC-2001",
    "age": 45,
    "region": "Nairobi",
    "cyst_size": 4.2
  },
  "observation_type": "cyst_size"
}
#Condition
{
  "patient_data": { "patient_id": "OC-2001" },
  "prediction_result": { "prediction": "Medication" }
}
#Care Plan
{
  "patient_data": { "patient_id": "OC-2001" },
  "care_template": {
    "ai_recommendation": { "treatment_plan": "Medication" },
    "treatment_protocol": { "duration": "3 months" }
  }
}

📁 File Structure
├── fhir_integration.py         # Main FHIR integration module (refactored)
├── fhir_api.py                #api testing
└── README_fhir_integration.md                   # This file
