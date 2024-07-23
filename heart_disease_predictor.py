import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
heart_data = pd.read_csv(r'heart_disease_data.csv')

# Prepare the data
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Define the function to make predictions
def predict_heart_disease():
    try:
        input_data = [float(entry.get()) for entry in entries]
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        result = "The Person has Heart Disease" if prediction[0] == 1 else "The Person does not have a Heart Disease"
        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")

# Create the main window
root = tk.Tk()
root.title("Heart Disease Predictor")
root.geometry("500x500")
root.configure(bg='#f0f0f0')

# Create and pack the heading label
heading = tk.Label(root, text="Heart Disease Predictor", font=("Helvetica", 16, "bold"), bg='#f0f0f0')
heading.pack(pady=10)

# Create and pack the input fields
frame = tk.Frame(root, bg='#f0f0f0')
frame.pack(pady=10)

labels = ['Age', 'Sex', 'Chest Pain Type', 'Blood Pressure', 'Cholesterol', 'Fasting Blood Sugar', 'Resting Electrocardiographic Results', 'Max Heart Rate', 'Exercise Induced Angina', 'Oldpeak Depression', 'Slope of Peak Exercise ST Segment', 'Number of Major Vessels', 'Thalassemia']
entries = []

for label in labels:
    row = tk.Frame(frame, bg='#f0f0f0')
    row.pack(pady=2)
    lbl = tk.Label(row, text=label, width=25, anchor='w', bg='#f0f0f0')
    lbl.pack(side='left')
    ent = tk.Entry(row)
    ent.pack(side='right')
    entries.append(ent)

# Create and pack the predict button
predict_button = tk.Button(root, text="Predict", command=predict_heart_disease, font=("Helvetica", 12), bg='#4CAF50', fg='white', width=15)
predict_button.pack(pady=20)

# Run the application
root.mainloop()
