🧬 Ovarian Cyst Behavior & Treatment Recommendation System
🚀 Project Overview
This AI-powered clinical decision support tool uses real patient data to:

Predict ovarian cyst behavior (Stable vs Unstable)

Recommend guideline-aligned management plans (e.g., Surgery, Observation)

Flag mismatches between clinical guidelines and model predictions

Developed during a 25-day hackathon, this solution is designed for low-data environments like Kenyan hospitals and clinics.

🔍 Features & Data Used
Inputs: Age, Menopause status, Cyst size, CA-125 level, Symptoms, Ultrasound summary

Engineered Features:

High CA-125 × Postmenopausal

Large Cyst × High CA-125

Symptom Severity (≥2 symptoms)

🧠 Model Summary
Algorithm: Decision Tree (max depth = 3) with SMOTE balancing

Task: Binary classification of cyst behavior

Outputs: Class prediction + probability score

Performance: Macro F1 ≈ 0.60 (Unstable detection prioritized)

📜 Guideline Logic for Management
Rules aligned with simplified Kenyan national clinical guidelines:

Clinical Pattern	Recommended Management
Postmenopausal + Cyst > 5cm + High CA-125	Surgery
Small cyst < 4cm + Normal CA-125	Observation
≥2 symptoms + High CA-125	Medication
Other combinations	Review

System flags any conflict between model and guideline-based logic.

📤 Outputs in Excel
You’ll find the following in ovarian_predictions_final.xlsx:

Predicted Cyst Behavior

Prediction Confidence

Rule-Based Recommendation

Clinical Flag (Mismatch or OK)

👩‍⚕️ Impact
Designed to assist clinicians in:

Early triage of risky cysts

Transparent, rule-aligned treatment planning

Handling medical data scarcity with AI augmentation

Built by Wendy Wanjiru 🇰🇪
Powered by Python, scikit-learn, and SMOTE