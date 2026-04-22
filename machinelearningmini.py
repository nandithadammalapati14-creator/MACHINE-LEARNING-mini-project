import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- STYLE ---------------- #
plt.style.use('dark_background')
sns.set(font_scale=1.2)

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv("dataset.csv.txt")

print("\n🔷 DATASET OVERVIEW")
print(data.head())

# ---------------- FEATURES ---------------- #
X = data.drop("doctor", axis=1)
y = data["doctor"]

# ---------------- SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ---------------- #
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 MODEL PERFORMANCE")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- GRAPH 1: DOCTOR DISTRIBUTION ---------------- #
plt.figure(figsize=(8,5))
sns.countplot(x='doctor', data=data)
plt.title("Doctor Distribution", fontsize=14)
plt.xlabel("Doctor Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("1_doctor_distribution.png")
plt.show()

# ---------------- GRAPH 2: PIE CHART ---------------- #
plt.figure(figsize=(6,6))
data['doctor'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', startangle=140
)
plt.title("Doctor Category Share")
plt.ylabel("")
plt.tight_layout()
plt.savefig("2_pie_chart.png")
plt.show()

# ---------------- GRAPH 3: HEATMAP ---------------- #
plt.figure(figsize=(8,6))
sns.heatmap(data.drop("doctor", axis=1).corr(), annot=True, cmap="coolwarm")
plt.title("Symptom Correlation Heatmap")
plt.tight_layout()
plt.savefig("3_heatmap.png")
plt.show()

# ---------------- GRAPH 4: CONFUSION MATRIX ---------------- #
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="viridis",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("4_confusion_matrix.png")
plt.show()

# ---------------- GRAPH 5: FEATURE IMPORTANCE (Approx) ---------------- #
# Using variance as simple importance indicator
importance = X.var().sort_values(ascending=False)

plt.figure(figsize=(8,5))
importance.plot(kind='bar')
plt.title("Feature Importance (Variance Based)")
plt.xlabel("Symptoms")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("5_feature_importance.png")
plt.show()

# ---------------- GRAPH 6: MODEL ACCURACY BAR ---------------- #
plt.figure(figsize=(5,5))
plt.bar(["Naive Bayes"], [accuracy * 100])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.savefig("6_accuracy.png")
plt.show()

# ---------------- USER PREDICTION ---------------- #
def predict_doctor():
    print("\n🔍 ENTER SYMPTOMS (1 = YES, 0 = NO)")

    fever = int(input("Fever: "))
    cough = int(input("Cough: "))
    headache = int(input("Headache: "))
    chest_pain = int(input("Chest Pain: "))
    skin_rash = int(input("Skin Rash: "))
    fatigue = int(input("Fatigue: "))

    symptoms = [[fever, cough, headache, chest_pain, skin_rash, fatigue]]

    prediction = model.predict(symptoms)

    print("\n✅ RECOMMENDED DOCTOR:", prediction[0])

# Run prediction
predict_doctor()