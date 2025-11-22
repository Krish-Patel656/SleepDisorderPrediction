import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


file_path = "Data.csv" 
df = pd.read_csv(file_path)


df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')

df["Sleep Disorder"] = df["Sleep Disorder"].astype(str).str.strip().str.title()

df = df[df["Sleep Disorder"].isin(["None", "Sleep Apnea"])]

print("Filtered data size:", len(df))
print(df["Sleep Disorder"].value_counts())

df = df.dropna(subset=["Sleep Disorder"])

df = df[df["Sleep Disorder"].isin(["None", "Sleep Apnea"])]

print("Filtered data size:", len(df))
print(df["Sleep Disorder"].value_counts())

X = df[["Stress Level", "Sleep Duration", "Quality of Sleep"]]
y = df["Sleep Disorder"]

y = y.map({"None": 0, "Sleep Apnea": 1})



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))



def predict_sleep_disorder():
    stress_level = float(input("Enter your Stress Level on a scale of 1-10: "))
    sleep_duration = float(input("Enter your Sleep Duration in hours: "))
    quality_sleep = float(input("Enter your Quality of Sleep on a scale of 1-10: "))

    user_input_df = pd.DataFrame(
        [[stress_level, sleep_duration, quality_sleep]],
        columns=["Stress Level", "Sleep Duration", "Quality of Sleep"]
    )

    user_input_scaled = scaler.transform(user_input_df)

    prediction = model.predict(user_input_scaled)
    prediction_prob = model.predict_proba(user_input_scaled)[0]

    labels = ["Normal", "Sleep Apnea"]
    print(f"\nPredicted Sleep Disorder: {labels[prediction[0]]}")
    print(f"Prediction Probability: {prediction_prob[prediction[0]]:.2f}")

predict_sleep_disorder()
