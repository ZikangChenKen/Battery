import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import os

WINDOW_SIZE = 0.5
Y_HOURS = 2.0      
TEST_SIZE = 0.2     
RANDOM_STATE = 42   

def load_data(root_path):
    failure_df = pd.read_csv(os.path.join(root_path, 'Battery_Failure_Time.csv'))
    failure_dict = dict(zip(failure_df['Test'], failure_df['Failure (hr)']))
    
    all_features = []
    all_clf_labels = []
    all_reg_labels = []
    
    for exp_id in failure_df['Test']:
        exp_path = os.path.join(root_path, str(exp_id), 'pressure_log.csv')
        
        try:
            df = pd.read_csv(exp_path)
            t = df['t(hr)'].values
            P = df['P'].values
            failure_time = failure_dict[exp_id]
        except (FileNotFoundError, KeyError):
            continue

        window_start = 0.0
        while window_start < failure_time:  
            window_end = window_start + WINDOW_SIZE
            
            mask = (t >= window_start) & (t < window_end)
            if sum(mask) < 1:  
                window_start += WINDOW_SIZE
                continue
            
            features = {
                'mean': np.mean(P[mask]),
                'std': np.std(P[mask]),
                'variance': np.var(P[mask]),
                'min': np.min(P[mask]),
                'max': np.max(P[mask]),
            }
            
            if sum(mask) >= 2:
                X = t[mask].reshape(-1, 1)
                y = P[mask]
                features['slope'] = LinearRegression().fit(X, y).coef_[0]
            else:
                features['slope'] = 0
            
            time_remaining = failure_time - window_end
            if time_remaining <= 0:
                window_start += WINDOW_SIZE
                continue 
            
            all_features.append(features)
            all_clf_labels.append(1 if time_remaining <= Y_HOURS else 0)
            all_reg_labels.append(time_remaining)
            
            window_start += WINDOW_SIZE
    
    return pd.DataFrame(all_features), np.array(all_clf_labels), np.array(all_reg_labels)

def main():
    path = r"C:\Users\chenz\OneDrive\桌面\Battery"
    features_df, clf_labels, reg_labels = load_data(path)
    
    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        features_df, clf_labels, reg_labels, 
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X_train, y_clf_train)
    clf_pred = clf.predict(X_test)
    print(f"Classification Accuracy: {accuracy_score(y_clf_test, clf_pred):.4f}")
    
    reg_mask = (y_reg_train <= Y_HOURS)
    reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    reg.fit(X_train[reg_mask], y_reg_train[reg_mask])
    reg_pred = reg.predict(X_test)
    print(f"Regression MSE: {mean_squared_error(y_reg_test, reg_pred):.4f}")

if __name__ == "__main__":
    main()