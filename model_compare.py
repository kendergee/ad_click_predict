from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

def preprocessing():
    # Load the cleaned dataset
    datasets = pd.read_csv('dataset/ad_click_dataset.csv')
    datasets = datasets.drop(['id','full_name'],axis=1)

    # Fill missing values
    for columns in datasets.columns:
        if datasets[columns].dtype == 'object':
            datasets[columns].fillna('missing',inplace=True)
        else:
            datasets[columns].fillna(datasets[columns].mean(),inplace=True)

    # 將年齡進行分段處理，並將 'missing' 視為一個段
    # bins = [0, 20, 30, 40, 50, 60, 100]
    # labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '60+']
    # datasets['age_binned'] = pd.cut(datasets['age'], bins=bins, labels=labels, right=False)
    # datasets['age_binned'] = datasets['age_binned'].cat.add_categories(['missing']).fillna('missing')
    # datasets = datasets.drop('age',axis=1)

    le = LabelEncoder()
    le_columns = ['gender','device_type','ad_position','browsing_history','time_of_day']
    for columns in le_columns:
        datasets[columns] = le.fit_transform(datasets[columns])

    X = datasets.drop('click',axis=1)
    y = datasets['click']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0) 

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    sm = SMOTE(random_state=0)
    X_train,y_train = sm.fit_resample(X_train,y_train)
    print('Data preprocessing done!')
    

    return X_train,X_test,y_train,y_test

def model_compare(X_train,X_test,y_train,y_test):
    from collections import Counter

# 計算正負類別的比例
    counter = Counter(y_train)
    class_weight_ratio = counter[0] / counter[1]   

    cat_features =['gender','device_type','ad_position','browsing_history','time_of_day'] 
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(),
         'XGBoost': xgb.XGBClassifier(),
        'LogisticRegression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        #  'CatBoost': CatBoostClassifier()
    }

    param_grids = {
        'RandomForest': {
            'class_weight': ['balanced', 'balanced_subsample'],
            'n_estimators': [200, 300, 400,450, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        },
        'XGBoost': {
            'scale_pos_weight': [class_weight_ratio],
            'n_estimators': [300, 400, 500,1000],
            'learning_rate': [0.01,0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.6,0.7, 0.8],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        # 'CatBoost': {
        #   'iterations': [500],
        #   'learning_rate': [0.1],
        #   'depth': [6]
        }

    best_models = {}
    results = []

    for name,model in models.items():
        print(f'Training {name}...')
        grid_search = GridSearchCV(model,param_grids[name],cv=5,n_jobs=-1,scoring='accuracy')
        grid_search.fit(X_train,y_train)
        best_models[name] = grid_search.best_estimator_
        print(f'{name} training done!')

        cv_scores = cross_val_score(best_models[name], X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f'{name} CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}')


        predictions = best_models[name].predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'{name} Test Accuracy: {accuracy:.4f}')
        report = classification_report(y_test, predictions, output_dict=True)
        confusion = confusion_matrix(y_test, predictions)
        print(accuracy)
        print(confusion)


        results.append({
            'model': name,
            'best_params': grid_search.best_params_,
            'cv_accuracy': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'test_accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion.tolist()
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_comparison_results_mean_weighted.csv',index=False)   
    
    print('Model comparison done!')
X_train,X_test,y_train,y_test = preprocessing()
model_compare(X_train,X_test,y_train,y_test)