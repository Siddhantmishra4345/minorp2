import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from data_loader import get_splits

def train_and_save():
    os.makedirs('models', exist_ok=True)
    print("Loading data...")
    X_train, y_train, X_cal, y_cal, X_eval, y_eval = get_splits()
    
    if X_train is None:
        print("Data not found. Cannot train models.")
        return
        
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42,
                                   multi_class='multinomial', solver='lbfgs')
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'models/lr_model.joblib')
    
    print("Training XGBoost...")
    le = LabelEncoder()
    y_train_xgb = le.fit_transform(y_train)
    y_cal_xgb   = le.transform(y_cal)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1, tree_method='hist'
    )
    xgb_model.fit(X_train, y_train_xgb)
    
    print("Calibrating XGBoost...")
    xgb_isotonic = CalibratedClassifierCV(xgb_model, cv='prefit', method='isotonic')
    xgb_isotonic.fit(X_cal, y_cal_xgb)
    
    joblib.dump(xgb_isotonic, 'models/xgb_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    print("Models saved successfully in models/")

if __name__ == "__main__":
    train_and_save()
