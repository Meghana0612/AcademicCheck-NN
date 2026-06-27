# AcademicCheck NN

AI-powered plagiarism detection system using NLP and Transformer models.

## Features

- Semantic plagiarism detection using BERT CrossEncoder
- Explainable AI using SHAP & LIME
- PDF, TXT and DOCX support
- Multi-document N×N comparison
- Pairwise similarity heatmap
- Sentence-level plagiarism analysis
- Automatic PDF report generation
- File vs Website plagiarism checking
- Multi-language document support
- Interactive Streamlit dashboard

## Tech Stack

- Python
- Streamlit
- spaCy
- Sentence Transformers
- Scikit-learn
- PyTorch

## Performance

- Accuracy: 84.27%
- F1 Score: 85.18%
- AUC: 92.01%

## Screenshots

### Home Page
![Home](screenshots/home.png)

### Result Page
![Result](screenshots/result.png)

### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

### ROC Curve
![ROC Curve](screenshots/roc_curve.png)

## Run

pip install -r requirements.txt

streamlit run app.py
