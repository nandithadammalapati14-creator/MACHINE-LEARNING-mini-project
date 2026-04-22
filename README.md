🩺 Symptom-Based Doctor Recommendation System Using Naive Bayes

📌 #Overview

This project focuses on recommending the most appropriate doctor based on patient symptoms using Machine Learning techniques. The system analyzes symptoms such as fever, cough, headache, chest pain, skin rash, and fatigue to predict the suitable medical specialist.

The system is built using the Naive Bayes Algorithm, which provides fast and efficient classification. It helps users quickly identify the right doctor, reducing confusion and saving time.

🚀 #Features
Symptom-based doctor prediction
Data preprocessing and feature selection
Implementation of Naive Bayes model
High accuracy prediction system
Performance evaluation using metrics
Data visualization using graphs:
Doctor distribution (bar chart)
Pie chart
Correlation heatmap
Confusion matrix
Feature importance graph

📂 #Dataset

Name: Symptom-Based Doctor Dataset
Type: Tabular Data
Source: Synthetic (created using Python, inspired by Kaggle datasets)
Features
Fever
Cough
Headache
Chest Pain
Skin Rash
Fatigue
Target Variable
Doctor Type:
General
Cardiologist
Dermatologist
Neurologist

⚙️# Technologies Used

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
🧠 Machine Learning Model
Naive Bayes (GaussianNB)
🔧 Model Workflow
Data Collection and Preprocessing
Feature Selection
Train-Test Split (80:20)
Model Training using Naive Bayes
Prediction of doctor specialization
Performance Evaluation
Visualization and Analysis
📊 Model Performance
Accuracy: ~100% (on given dataset)

✔ The model correctly predicts doctor types based on symptoms
✔ Fast and efficient performance

📈 Output Visualizations

The project generates the following outputs:

Doctor Distribution Graph
Pie Chart
Symptom Correlation Heatmap
Confusion Matrix
Feature Importance Graph
Accuracy Graph
📁 Project Structure
Symptom-Doctor-Recommendation/
│
├── dataset.csv
├── main.py
├── output/
│   ├── bar_chart.png
│   ├── pie_chart.png
│   ├── heatmap.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── README.md
▶️ How to Run the Project
1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
2. Run the program
python main.py
📌 Results Summary

The system successfully recommends the appropriate doctor based on symptoms with high accuracy. The Naive Bayes algorithm performs efficiently for classification tasks. Symptoms like fever and headache play a key role in prediction. The model is simple, fast, and suitable for real-time applications.

🔮 Future Improvements
Use real-world datasets from Kaggle
Add more symptoms and doctor categories
Develop a web or mobile application
Integrate real hospital and doctor data
Apply advanced machine learning models
🤝 Contributing

Contributions are welcome. Feel free to fork this repository and improve the project.

📜 License

This project is for academic and educational purposes only.

👩‍💻 Author

**Nanditha Dammalapati**
