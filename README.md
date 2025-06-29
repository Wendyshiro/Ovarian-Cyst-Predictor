🧬 Ovarian Cyst Predictor & Treatment Recommendation System

This AI-powered tool helps clinicians and patients make informed decisions about ovarian cysts. It predicts cyst behavior and recommends treatment pathways based on clinical data and evidence-based guidelines.

## 🚀 Project Goals

- Predict the likelihood of cyst growth or malignancy using ML.
- Recommend treatment plans: observation, medication, or surgery.
- Export downloadable Excel reports with prediction confidence and clinical flags.
- Future features: real-time medication & surgical tool tracking, cost estimations, and financing support.

## 🛠️ Tech Stack

- Python (Pandas, scikit-learn, NumPy)
- ExcelWriter for data export
- Git & GitHub for version control
- (Planned) Streamlit / Flask for deployment
- Currently used Descion Trees ML model


## 📊 Sample Output (Excel Report)

Below is a sample of the model's output, highlighting predictions and treatment recommendations for ovarian cyst cases.

| Patient ID | Age | Cyst Size (cm) | CA-125 | Symptoms                         | Predicted Behavior | Recommended Treatment | Confidence | Clinical Flag |
|------------|-----|----------------|--------|----------------------------------|---------------------|------------------------|------------|----------------|
| OC-1000    | 52  | 3.2            | 19     | Pelvic pain, Nausea, Bloating    | Unstable            | Observation            | 0.53       | OK             |
| OC-1001    | 62  | 7.9            | 111    | Bloating                         | Unstable            | Surgery                | 0.68       | OK             |
| OC-1002    | 59  | 2.2            | 123    | Pelvic pain, Irregular periods   | Unstable            | Medication             | 0.53       | OK             |
| OC-1003    | 64  | 5.5            | 116    | Nausea, Irregular periods        | Unstable            | Surgery                | 1.00       | OK             |


## ⚙️ How It Works

1. Input patient data: age, menopausal status, cyst size, symptoms, etc.
2. The model classifies risk and outputs a prediction.
3. Recommender system suggests next clinical steps.
4. Results are saved to Excel with risk flags and confidence scores.

## 📈 Roadmap

- [x] Predict cyst behavior
- [x] Generate downloadable treatment report
- [ ] Add tracking of medication and surgical tool availability
- [ ] Include cost estimations and financing options
- [ ] Build interactive web-based interface (Streamlit/Flask)

## 📄 License

MIT License — free to use and modify with attribution.

## 👩🏽‍💻 Developed by Wendy Wanjiru

MSc Software Engineering | AI for Healthcare Advocate

Connect on [LinkedIn](https://www.linkedin.com/in/wendy-waweru18/)
