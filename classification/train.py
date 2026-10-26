import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def train_and_evaluate(data_dir="data/processed", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    # Map labels
    label_map = {'FAKE': 0, 'REAL': 1}
    y_train = train_df['label'].map(label_map)
    y_val = val_df['label'].map(label_map)
    y_test = test_df['label'].map(label_map)
    
    # Feature extraction
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['full_text'])
    X_val = vectorizer.transform(val_df['full_text'])
    X_test = vectorizer.transform(test_df['full_text'])
    
    candidate_models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(dual=False, random_state=42))
    }
    
    best_f1 = 0.0
    best_model_name = None
    best_model = None
    
    print("=== Validation Set Comparison ===")
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average='macro')
        print(f"Model: {name} | Validation Macro F1: {score:.4f}")
        
        if score > best_f1:
            best_f1 = score
            best_model_name = name
            best_model = model

    print(f"\nSelected Model: {best_model_name}")
    
    # Final evaluation on untouched Test Set
    test_preds = best_model.predict(X_test)
    print("\n=== Untouched Final Test Set Evaluation ===")
    print(classification_report(y_test, test_preds, target_names=['FAKE', 'REAL']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
    
    # Save artifacts
    joblib.dump(best_model, os.path.join(model_dir, "fake_news_model.joblib"))
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.joblib"))
    print("\nArtifacts successfully saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_evaluate()