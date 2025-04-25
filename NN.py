import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Global parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
PRE_FAILURE_HOURS = 5

# Neural Network Models --------------------------------------------------------
def build_classifier(input_dim):
    """Classification model for failure prediction"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_regressor(input_dim):
    """Regression model for time remaining prediction""" 
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def load_data(excel_path, WINDOW_SIZE, Y_HOURS):
    xls = pd.ExcelFile(excel_path)
    all_features = []
    all_clf_labels = []
    all_reg_labels = []

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            voltage_df = df.iloc[:, [0, 1]].dropna().reset_index(drop=True)
            pressure_df = df.iloc[:, [2, 3]].dropna().reset_index(drop=True)
            
            voltage_df.columns = ['time/h', '<Ewe>/V']
            pressure_df.columns = ['t(hr)', 'P']
            
            failure_time = min(
                voltage_df['time/h'].max(), 
                pressure_df['t(hr)'].max()
            )
            
            voltage_data = voltage_df[
                (voltage_df['time/h'] >= (failure_time - PRE_FAILURE_HOURS)) &
                (voltage_df['time/h'] <= failure_time)
            ].copy()
            
            pressure_data = pressure_df[
                (pressure_df['t(hr)'] >= (failure_time - PRE_FAILURE_HOURS)) &
                (pressure_df['t(hr)'] <= failure_time)
            ].copy()
            
            v_initial = voltage_df.iloc[0]['<Ewe>/V']
            p_initial = pressure_df.iloc[0]['P']
            
            voltage_data['<Ewe>/V_pct'] = (
                (voltage_data['<Ewe>/V'] - v_initial) / v_initial
            )
            pressure_data['P_pct'] = (
                (pressure_data['P'] - p_initial) / p_initial
            )
            
            min_time = max(voltage_data['time/h'].min(), pressure_data['t(hr)'].min())
            max_time = failure_time
            window_start = min_time
            
            while window_start < max_time:
                window_end = window_start + WINDOW_SIZE
                
                voltage_window = voltage_data[
                    (voltage_data['time/h'] >= window_start) &
                    (voltage_data['time/h'] < window_end)
                ]
                v_features = _extract_features(voltage_window['<Ewe>/V_pct'].values, "voltage_")
                
                pressure_window = pressure_data[
                    (pressure_data['t(hr)'] >= window_start) &
                    (pressure_data['t(hr)'] < window_end)
                ]
                p_features = _extract_features(pressure_window['P_pct'].values, "pressure_")
                
                combined_features = {**v_features, **p_features}
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
                
        except Exception as e:
            print(f"Errors occurred when processing {sheet_name}: {str(e)}")
            continue
    
    return pd.DataFrame(all_features), np.array(all_clf_labels), np.array(all_reg_labels)

# Function to extract features from the data
def _extract_features(values, prefix):
    features = {}
    if len(values) == 0:
        return features
    
    features.update({
        f'{prefix}mean': np.mean(values),
        f'{prefix}std': np.std(values),
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

# Main Execution ---------------------------------------------------------------
def main():
    excel_path = r"C:\Users\chenz\OneDrive\桌面\Battery\Down-selected tests.xlsx"
    
    # 初始化输出目录（保持原有目录结构不变）
    output_dirs = {
        'results': os.path.join(os.path.dirname(excel_path), 'results')
    }
    os.makedirs(output_dirs['results'], exist_ok=True)

    window_sizes = [0.5, 0.3, 0.1, 0.05, 0.01]
    y_hours_list = [1, 2]
    
    results = []
    
    for Y_HOURS in y_hours_list:
        for ws in window_sizes:
            print(f"\n=== Processing Y_HOURS={Y_HOURS}h, Window={ws}h ===")
            
            # 加载数据
            features_df, clf_labels, reg_labels = load_data(excel_path, ws, Y_HOURS)
            if len(features_df) == 0:
                continue
                
            # 数据分割
            X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
                features_df.values, clf_labels, reg_labels.flatten(), 
                test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            
            # 分类模型训练
            clf_model = build_classifier(X_train.shape[1])
            early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
            clf_model.fit(
                X_train, y_clf_train,
                validation_split=0.2,
                epochs=200,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            # 分类评估
            y_clf_pred = (clf_model.predict(X_test) > 0.5).astype(int)
            clf_acc = accuracy_score(y_clf_test, y_clf_pred)
            
            # 回归模型训练
            reg_mask = (y_reg_train <= Y_HOURS)
            reg_mse = np.nan
            
            if sum(reg_mask) > 1:
                reg_model = build_regressor(X_train.shape[1])
                reg_model.fit(
                    X_train[reg_mask], y_reg_train[reg_mask],
                    validation_split=0.2,
                    epochs=200,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0
                )
                y_reg_pred = reg_model.predict(X_test).flatten()
                reg_mse = mean_squared_error(y_reg_test, y_reg_pred)
            
            # 存储结果
            results.append({
                'y_hours': Y_HOURS,
                'window_size': ws,
                'accuracy': clf_acc,
                'mse': reg_mse
            })
            
            # 保存简化结果图
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            safe_y_test = np.asarray(y_reg_test).flatten()
            safe_y_pred = np.asarray(y_reg_pred).flatten() if not np.isnan(reg_mse) else []
            sns.scatterplot(x=safe_y_test, y=safe_y_pred)
            plt.title(f'Regression MSE: {reg_mse:.2f}')
            
            plt.subplot(1,2,2)
            sns.heatmap(pd.DataFrame({'True':y_clf_test, 'Pred':y_clf_pred.flatten()}).corr(), 
                        annot=True, fmt=".2f")
            plt.title(f'Classification Acc: {clf_acc:.2f}')
            
            plt.savefig(os.path.join(output_dirs['results'], f'results_Y{Y_HOURS}_W{ws}.png'))
            plt.close()

    # 最终结果输出
    print("\n=== Final Results ===")
    print("{:<12} {:<12} {:<12} {:<12}".format("Y_Hours", "Window(h)", "Accuracy", "MSE"))
    for res in results:
        mse = f"{res['mse']:.4f}" if not np.isnan(res['mse']) else "N/A"
        print(f"{res['y_hours']:<12} {res['window_size']:<12.2f} {res['accuracy']:<12.4f} {mse:<12}")

if __name__ == "__main__":
    main()