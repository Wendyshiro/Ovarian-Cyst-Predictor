🧬 Ovarian Cyst Predictor & Treatment Recommendation System

An AI-powered clinical decision support tool that predicts ovarian cyst behavior and recommends treatment pathways using advanced machine learning techniques. Built for early detection and support in PCOS-related and gynecological care.
🚀 Project Goals

    Accurately predict the likelihood of cyst growth or malignancy using ML models trained on clinical and demographic data.

    Recommend treatment plans (observation, medication, or surgery) aligned with evidence-based guidelines.

    Export downloadable Excel reports with prediction confidence and clinical red flags.

    Future features include real-time medication/surgical inventory tracking, cost estimation, and financing support.

🧠 Recent Milestone

    ✅ Achieved 90% prediction accuracy using Random Forest and XGBoost classifiers, improving performance and reliability over the initial Decision Tree model.

🛠️ Tech Stack

    Python: Pandas, scikit-learn, NumPy, XGBoost

    Modeling: Random Forest, XGBoost (with hyperparameter tuning)

    Reporting: ExcelWriter for generating clinical reports

    Deployment: Streamlit (web-based demo)

    Version Control: Git & GitHub

📊 Sample Output (Excel Report)

The tool exports a rich Excel report combining clinical input data, derived features, and predictions from multiple machine learning models.
Below is a sample snapshot of key fields:
| Patient ID | Age | Cyst Size (cm) | CA-125 | Symptoms                       | Predicted Behavior | Recommended Treatment | Confidence | Clinical Flag |
| ---------- | --- | -------------- | ------ | ------------------------------ | ------------------ | --------------------- | ---------- | ------------- |
| OC-1000    | 52  | 3.2            | 19     | Pelvic pain, Nausea, Bloating  | Unstable           | Observation           | 0.87       | OK            |
| OC-1001    | 62  | 7.9            | 111    | Bloating                       | Unstable           | Surgery               | 0.91       | OK            |
| OC-1002    | 59  | 2.2            | 123    | Pelvic pain, Irregular periods | Unstable           | Medication            | 0.84       | OK            |
| OC-1003    | 64  | 5.5            | 116    | Nausea, Irregular periods      | Unstable           | Surgery               | 0.95       | OK            |

🧠 Model Architecture & Prediction Logic

    Feature Engineering includes clinical ratios like CA125_Size_Ratio, Symptom_Age_Ratio, and binary flags for post-menopausal risk.

    Random Forest & XGBoost Models both generate predictions with probability scores.

    Ensemble Logic:

        If both models agree → prediction adopted

        If models differ → select based on highest confidence score

    Treatment Recommendation is rule-based and layered over the model output.

✅ Current Model Accuracy: 90%, validated on structured clinical cases.
🧪 Possible Next Steps

    Incorporate patient history and hormonal profiles for deeper personalization

    Add NLP pipeline for symptom description input

    Partner with clinicians for field testing and feedback

📄 License

MIT License — Open source and free to use or adapt with attribution.
👩🏽‍💻 Developed by Wendy Wanjiru

MSc Software Engineering | AI for Healthcare Advocate

🔗 Connect on LinkedIn: https://www.linkedin.com/in/wendy-waweru18/
