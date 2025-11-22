# Sleep Disorder Prediction

**Predicting Sleep Apnea using AI and Machine Learning**

---

## About This Project

This is a small machine learning project that predicts whether someone has **Sleep Apnea** based on a few key indicators:  

- **Stress Level** (1-10 scale)  
- **Sleep Duration** (in hours)  
- **Quality of Sleep** (1-10 scale)  

We use a **Random Forest Classifier** in Python, trained on labeled data, achieving roughly **85% accuracy**. Users can input their own values to get predictions in real-time.

> This project is **for educational purposes only**. It is not a medical diagnostic tool.

---

## Features

- Predict Sleep Apnea or Normal sleep status based on user input.  
- Interactive command-line interface for real-time predictions.  
- Trained using scikit-learn’s Random Forest for robust classification.  

---

## Technologies Used

- Python
- pandas  
- scikit-learn  

---

## How to Use

### 1. Clone the repository

```bash
git clone https://github.com/Krish-Patel656/SleepDisorderPrediction.git
cd SleepDisorderPrediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the predictor
```bash
python predict_sleep_disorder.py
```