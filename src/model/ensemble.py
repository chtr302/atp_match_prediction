import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, log_loss

class EnsembleTrainer:
    def __init__(self, trained_models):
        self.models = trained_models
        self.results = []

    def soft_voting(self, X_test, y_test):
        print("\n--- ENSEMBLE: SOFT VOTING ---")
        probs = []
        valid_models = []
        
        # Duyệt qua tất cả model có trong danh sách
        for name, model in self.models.items():
            try:
                p = model.predict_proba(X_test)[:, 1]
                probs.append(p)
                valid_models.append(name)
            except AttributeError:
                print(f"Bỏ qua {name} (không có predict_proba)")
        
        print(f"Đang vote dựa trên: {', '.join(valid_models)}")
        
        if not probs:
            return None
            
        y_prob_avg = np.mean(probs, axis=0)
        y_pred_avg = (y_prob_avg >= 0.5).astype(int)

        res = {
            'Model': 'Ensemble Soft Voting',
            'Accuracy': accuracy_score(y_test, y_pred_avg),
            'Precision': precision_score(y_test, y_pred_avg),
            'F1-score': f1_score(y_test, y_pred_avg),
            'ROC-AUC': roc_auc_score(y_test, y_prob_avg),
            'Log Loss': log_loss(y_test, y_prob_avg)
        }
        self.results.append(res)
        print(f"  -> Accuracy: {res['Accuracy']:.4f}, AUC: {res['ROC-AUC']:.4f}")
        return res

    def stacking(self, X_train, y_train, X_test, y_test):
        print("\n--- ENSEMBLE: STACKING ---")
        meta_train_list = []
        meta_test_list = []
        valid_models = []
        
        for name, model in self.models.items():
            try:
                # Cần ensure tính tương thích chiều dữ liệu
                train_prob = model.predict_proba(X_train)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]
                
                meta_train_list.append(train_prob)
                meta_test_list.append(test_prob)
                valid_models.append(name)
            except AttributeError:
                pass
            
        X_train_meta = np.column_stack(meta_train_list)
        X_test_meta = np.column_stack(meta_test_list)
        
        # Meta-model (Logistic Regression)
        meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        meta_model.fit(X_train_meta, y_train)
        
        y_prob_stack = meta_model.predict_proba(X_test_meta)[:, 1]
        y_pred_stack = (y_prob_stack >= 0.5).astype(int)
        
        res = {
            'Model': 'Ensemble Stacking',
            'Accuracy': accuracy_score(y_test, y_pred_stack),
            'Precision': precision_score(y_test, y_pred_stack),
            'F1-score': f1_score(y_test, y_pred_stack),
            'ROC-AUC': roc_auc_score(y_test, y_prob_stack),
            'Log Loss': log_loss(y_test, y_prob_stack)
        }
        self.results.append(res)
        print(f"  -> Accuracy: {res['Accuracy']:.4f}, AUC: {res['ROC-AUC']:.4f}")
        return res

    def get_results(self):
        return pd.DataFrame(self.results)
