import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import os
import matplotlib.pyplot as plt
import seaborn as sns


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

def compare_feature_distributions(group_0, group_1, features):
    comparison = []
    for feat in features:
        mean_0 = group_0[feat].mean()
        mean_1 = group_1[feat].mean()
        std_0 = group_0[feat].std()
        std_1 = group_1[feat].std()
        median_0 = group_0[feat].median()
        median_1 = group_1[feat].median()
        
        mean_diff_ratio = (mean_1 - mean_0) / (mean_0 + 1e-6)  # avoid division by zero
        
        comparison.append({
            'Feature': feat,
            'Mean (True=0)': mean_0,
            'Mean (True=1)': mean_1,
            'Mean Diff Ratio': mean_diff_ratio,
            'Std (True=0)': std_0,
            'Std (True=1)': std_1,
            'Median (True=0)': median_0,
            'Median (True=1)': median_1
        })
    
    return pd.DataFrame(comparison)

def plot_feature_distribution(feat, group_0, group_1, save_path=None):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x='clf_label_true', y=feat, data=pd.concat([group_0, group_1]))
    plt.title(f'Boxplot of {feat}')
    plt.xlabel('True Label')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(group_0[feat], label='True=0', fill=True)
    sns.kdeplot(group_1[feat], label='True=1', fill=True)
    plt.title(f'Density Plot of {feat}')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{feat}_distribution.png"), bbox_inches='tight')
    plt.close()

def plot_results(results_df, Y_HOURS, ws, clf, reg, X_train, save_dir):
    plt.figure(figsize=(15,10))
    
    plt.subplot(2,2,1)
    sns.scatterplot(x='reg_label_true', y='reg_label_pred', data=results_df)
    plt.plot([0, results_df['reg_label_true'].max()], [0, results_df['reg_label_true'].max()], 'r--')
    plt.title(f'Regression Results (Y={Y_HOURS}h, W={ws}h)')
    plt.xlabel('True Time Remaining')
    plt.ylabel('Predicted Time Remaining')
    
    plt.subplot(2,2,2)
    sns.countplot(x='clf_label_pred', hue='clf_label_true', data=results_df)
    plt.title(f'Classification Distribution (Y={Y_HOURS}h, W={ws}h)')
    
    plt.subplot(2,2,3)
    feature_importance = pd.Series(clf.feature_importances_, index=X_train.columns)
    feature_importance.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Classification Features')
    
    plt.subplot(2,2,4)
    feature_importance_reg = pd.Series(reg.feature_importances_, index=X_train.columns)
    feature_importance_reg.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Regression Features')
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, f'results_Y{Y_HOURS}_W{ws}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved results plot to {output_path}")

def main():
    battery_path = r"C:\Users\chenz\OneDrive\桌面\Battery"

    output_dirs = {
        'full_data': os.path.join(battery_path, 'full_data'),
        'feature_comp': os.path.join(battery_path, 'feature_comparison'),
        'results': os.path.join(battery_path, 'results'),
        'feat_dist': os.path.join(battery_path, 'feature_distributions')
    }

    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
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
            y_proba = clf.predict_proba(X_test)[:, 1]
            clf_acc = accuracy_score(y_clf_test, clf.predict(X_test))

            if len(np.unique(y_clf_test)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(y_clf_test, y_proba)
            
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
                'auc': auc, 
                'mse': reg_mse
            })

            full_df = features_df.copy()
            full_df['clf_label_true'] = clf_labels
            full_df['reg_label_true'] = reg_labels
            
            full_df['clf_label_pred'] = clf.predict(features_df.values)
            full_df['reg_label_pred'] = reg.predict(features_df.values)

            group_0 = full_df[full_df['clf_label_true'] == 0]  
            group_1 = full_df[full_df['clf_label_true'] == 1] 

            label_columns = ['clf_label_true', 'reg_label_true', 
                'clf_label_pred', 'reg_label_pred']
            feature_columns = [col for col in full_df.columns if col not in label_columns]

            comparison_df = compare_feature_distributions(group_0, group_1, feature_columns)
            comp_csv_path = os.path.join(
                output_dirs['feature_comp'],
                f"feature_comparison_Y{Y_HOURS}_W{ws}.csv"
            )
            comparison_df.to_csv(comp_csv_path, index=False)
            print(f"Saved feature comparison to {comp_csv_path}")

            output_dir = os.path.join(
                output_dirs['feat_dist'],
                f"Y{Y_HOURS}_W{ws}"
            )
            os.makedirs(output_dir, exist_ok=True)

            top_features = pd.Series(clf.feature_importances_, index=feature_columns)
            top_features = top_features.nlargest(5).index.tolist()

            for feat in top_features:
                plot_feature_distribution(feat, group_0, group_1, output_dir)
            print(f"Saved {len(top_features)} feature plots to {output_dir}")

            output_csv = os.path.join(
                output_dirs['full_data'], 
                f"full_data_Y{Y_HOURS}_W{ws}.csv"
            )
            full_df.to_csv(output_csv, index=False)
            print(f"Saved full data to {output_csv}")

            plot_results(full_df, Y_HOURS, ws, clf, reg, X_train, save_dir=output_dirs['results'])
            
            print(f"[Y={Y_HOURS}h][W={ws}h] Accuracy: {clf_acc:.4f} | AUC: {auc:.4f} | MSE: {reg_mse if not np.isnan(reg_mse) else 'N/A'}")

    print("\n\n=== Final Results (Grouped by Y_HOURS) ===")
    for yh in y_hours:
        print(f"\n--- Y_HOURS = {yh} hours ---")
        print("{:<12} {:<12} {:<12} {:<12}".format("Window(h)", "Accuracy", "AUC", "MSE"))
        yh_results = [r for r in results if r['y_hours'] == yh]
        for res in yh_results:
            mse = f"{res['mse']:.4f}" if not np.isnan(res['mse']) else "N/A    "
            auc = f"{res['auc']:.4f}" if not np.isnan(res['auc']) else "N/A    "
            print("{:<12.2f} {:<12.4f} {:<12} {:<12}".format(
                res['window_size'],
                res['accuracy'],
                auc,
                mse
            ))

if __name__ == "__main__":
    main()