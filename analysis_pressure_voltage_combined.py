import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import os


# WINDOW_SIZE = 1.0  
# Y_HOURS = 2.0     
TEST_SIZE = 0.2     
RANDOM_STATE = 42   

def load_multimodal_data(root_path, WINDOW_SIZE, Y_HOURS):
    failure_df = pd.read_csv(os.path.join(root_path, 'Battery_Failure_Time.csv'))
    failure_dict = dict(zip(failure_df['Test'], failure_df['Failure (hr)']))
    
    all_features = []
    all_clf_labels = []
    all_reg_labels = []

    for exp_id in failure_dict.keys():
        try:
            pressure_path = os.path.join(root_path, str(exp_id), 'pressure_log.csv')
            voltage_path = os.path.join(root_path, str(exp_id), 'voltage.csv')
            

            df_p = pd.read_csv(pressure_path)
            t_p = df_p['t(hr)'].values
            pressure = df_p['P'].values
            

            df_v = pd.read_csv(voltage_path)
            t_v = df_v['time/h'].values
            voltage = df_v['<Ewe>/V'].values
            
            failure_time = failure_dict[exp_id]
        except (FileNotFoundError, KeyError) as e:
            print(f"ignore {exp_id}: {str(e)}")
            continue

        min_time = max(t_p[0], t_v[0])
        max_time = min(t_p[-1], t_v[-1], failure_time)
        window_start = 0.0
        
        while window_start < max_time:
            window_end = window_start + WINDOW_SIZE
            if window_end > max_time:
                break

            mask_p = (t_p >= window_start) & (t_p < window_end)
            p_values = pressure[mask_p]
            p_features = _extract_features(p_values, "pressure_")
            
            mask_v = (t_v >= window_start) & (t_v < window_end)
            v_values = voltage[mask_v]
            v_features = _extract_features(v_values, "voltage_")
            
            
            combined_features = {**p_features, **v_features}
            if not combined_features:
                window_start += WINDOW_SIZE
                continue
            
           
            time_remaining = failure_time - window_end
            if time_remaining <= 0:
                window_start += WINDOW_SIZE
                continue
                
            all_features.append(combined_features)
            all_clf_labels.append(1 if time_remaining <= Y_HOURS else 0)
            all_reg_labels.append(time_remaining)
            
            window_start += WINDOW_SIZE
    
    return pd.DataFrame(all_features), np.array(all_clf_labels), np.array(all_reg_labels)

def _extract_features(values, prefix):
    features = {}
    if len(values) == 0:
        return features
    
    features.update({
        f'{prefix}mean': np.mean(values),
        f'{prefix}std': np.std(values),
        f'{prefix}variance': np.var(values),
        f'{prefix}min': np.min(values),
        f'{prefix}max': np.max(values),
    })
    
    if len(values) >= 2:
        X = np.arange(len(values)).reshape(-1, 1)
        try:
            slope = LinearRegression().fit(X, values).coef_[0]
        except:
            slope = 0
        features[f'{prefix}slope'] = slope
    else:
        features[f'{prefix}slope'] = 0
    
    return features

def main():
    battery_path = r"C:\Users\chenz\OneDrive\桌面\Battery"
    
    window_sizes = [1.8, 1.5, 1.2, 1, 0.6, 0.5, 0.3, 0.1, 0.05, 0.01]
    y_hours = [1, 2]
    
    results = []
    
    for Y_HOURS in y_hours:
        for ws in window_sizes:
            print(f"\n=== Current Y_HOURS: {Y_HOURS} h | Window Size: {ws} h ===")
            
            features_df, clf_labels, reg_labels = load_multimodal_data(battery_path, ws, Y_HOURS)
            
            if len(features_df) == 0:
                print(f"No valid Data for Y_HOURS={Y_HOURS}h, Window={ws}h")
                continue
            
            features_df = features_df.fillna(features_df.mean())
            
            X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
                features_df, clf_labels, reg_labels, 
                test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
            clf.fit(X_train, y_clf_train)
            clf_acc = accuracy_score(y_clf_test, clf.predict(X_test))
            
            reg_mask = (y_reg_train <= Y_HOURS)
            if sum(reg_mask) < 1: 
                reg_mse = np.nan
            else:
                reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
                reg.fit(X_train[reg_mask], y_reg_train[reg_mask])
                reg_mse = mean_squared_error(y_reg_test, reg.predict(X_test))
            
            results.append({
                'y_hours': Y_HOURS,
                'window_size': ws,
                'accuracy': clf_acc,
                'mse': reg_mse
            })
            
            print(f"[Y={Y_HOURS}h][W={ws}h] Accuracy: {clf_acc:.4f} | MSE: {reg_mse if not np.isnan(reg_mse) else 'N/A'}")

    print("\n\n=== Final Results (Grouped by Y_HOURS) ===")
    for yh in y_hours:
        print(f"\n--- Y_HOURS = {yh} hours ---")
        print("{:<12} {:<12} {:<12}".format("Window(h)", "Accuracy", "MSE"))
        yh_results = [r for r in results if r['y_hours'] == yh]
        for res in yh_results:
            mse = f"{res['mse']:.4f}" if not np.isnan(res['mse']) else "N/A    "
            print("{:<12.2f} {:<12.4f} {:<12}".format(
                res['window_size'],
                res['accuracy'],
                mse
            ))

if __name__ == "__main__":
    main()