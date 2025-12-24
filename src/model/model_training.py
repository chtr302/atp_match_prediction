import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, log_loss

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = []
        
        self.pipelines = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=self.random_state
                ))
            ]),
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=300,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]),
            'XGBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('model', XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric='logloss',
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('model', SVC(
                    kernel='rbf',         # Radial Basis Function (Phi tuyến)
                    C=1.0,                # Regularization (C càng nhỏ càng chống overfitting)
                    probability=True,     # Bắt buộc để dùng Soft Voting
                    class_weight='balanced',
                    random_state=self.random_state
                ))
            ])
        }

    def train_evaluate(self, X_train, y_train, X_test, y_test):
        print("\n--- HUẤN LUYỆN MÔ HÌNH CƠ SỞ (BASE MODELS) ---")
        
        for name, pipe in self.pipelines.items():
            print(f"Đang train {name}...")
            pipe.fit(X_train, y_train)
            self.models[name] = pipe
            
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            
            res = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'F1-score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_prob),
                'Log Loss': log_loss(y_test, y_prob)
            }
            self.results.append(res)
            print(f"  -> Accuracy: {res['Accuracy']:.4f}, AUC: {res['ROC-AUC']:.4f}")
            
        return pd.DataFrame(self.results)

    def get_trained_models(self):
        return self.models
