import tkinter as tk
from tkinter import messagebox
import joblib

# Load Model and Vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prediction Function
def predict_message():
    msg = text_box.get("1.0", tk.END).strip()

    if msg == "":
        messagebox.showwarning("Warning", "Please enter a message.")
        return

    msg_vector = vectorizer.transform([msg])

    prediction = model.predict(msg_vector)[0]
    probability = model.predict_proba(msg_vector)[0]

    ham = probability[0] * 100
    spam = probability[1] * 100

    if prediction == 1:
        result_label.config(
            text=f"🚨 SPAM MESSAGE\n\nSpam: {spam:.2f}%\nHam: {ham:.2f}%",
            fg="red"
        )
    else:
        result_label.config(
            text=f"✅ NOT SPAM\n\nHam: {ham:.2f}%\nSpam: {spam:.2f}%",
            fg="green"
        )

# Window
root = tk.Tk()
root.title("Smart Email Spam Detection")
root.geometry("600x450")
root.resizable(False, False)

# Heading
title = tk.Label(
    root,
    text="Smart Email Spam Detection System",
    font=("Arial", 18, "bold")
)
title.pack(pady=15)

# Text Box
text_box = tk.Text(root, height=8, width=60, font=("Arial", 11))
text_box.pack()

# Button
predict_btn = tk.Button(
    root,
    text="Predict",
    font=("Arial", 12, "bold"),
    command=predict_message,
    bg="#4CAF50",
    fg="white",
    width=15
)
predict_btn.pack(pady=15)

# Result
result_label = tk.Label(
    root,
    text="Enter a message and click Predict",
    font=("Arial", 13)
)
result_label.pack()

root.mainloop()