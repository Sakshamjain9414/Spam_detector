import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep Required Columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert Labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Features & Target
X = df['message']
y = df['label']

# ==========================
# Convert Text to Numbers
# ==========================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================
model = MultinomialNB()
model.fit(X_train, y_train)

# ==========================
# Save Model
# ==========================
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load Saved Model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================
accuracy = accuracy_score(y_test, y_pred)

print("="*50)
print(" SMART EMAIL SPAM DETECTION SYSTEM ")
print("="*50)

print(f"\nAccuracy : {accuracy*100:.2f}%")

# ==========================
# Classification Report
# ==========================
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==========================
# Confusion Matrix
# ==========================
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix\n")
print(cm)

# ==========================
# Top Spam Words
# ==========================
feature_names = vectorizer.get_feature_names_out()
spam_words = model.feature_log_prob_[1]

top = np.argsort(spam_words)[-15:]

print("\nTop Spam Words\n")

for i in top[::-1]:
    print(feature_names[i])

# ==========================
# User Prediction
# ==========================
print("\nType 'exit' to stop.")

while True:

    msg = input("\nEnter Message : ")

    if msg.lower() == "exit":
        print("\nProgram Closed.")
        break

    msg_vector = vectorizer.transform([msg])

    prediction = model.predict(msg_vector)[0]

    probability = model.predict_proba(msg_vector)[0]

    ham_prob = probability[0] * 100
    spam_prob = probability[1] * 100

    print("\n-----------------------------")

    if prediction == 1:
        print("Prediction : 🚨 SPAM MESSAGE")
    else:
        print("Prediction : ✅ NOT SPAM")

    print(f"Ham Probability  : {ham_prob:.2f}%")
    print(f"Spam Probability : {spam_prob:.2f}%")

    # Save Prediction History
    history = pd.DataFrame({
        "Message": [msg],
        "Prediction": ["Spam" if prediction == 1 else "Not Spam"],
        "Spam Probability": [f"{spam_prob:.2f}%"]
    })

    history.to_csv(
        "prediction_history.csv",
        mode="a",
        header=False,
        index=False
    )

    print("Prediction Saved Successfully.")