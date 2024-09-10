from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd

datasets = pd.read_csv('dataset/ad_click_dataset.csv')
datasets = datasets.drop(['id','full_name','age'],axis=1)

    # 填補缺失值
for column in datasets.columns:
    if datasets[column].dtype == 'object':
        datasets[column].fillna('missing', inplace=True)
    else:
        datasets[column].fillna(datasets[column].mean(), inplace=True)

    # 將年齡進行分段處理，並將 'missing' 視為一個段
# bins = [0, 20, 30, 40, 50, 60, 100]
# labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '60+']
# datasets['age_binned'] = pd.cut(datasets['age'], bins=bins, labels=labels, right=False)
# datasets['age_binned'] = datasets['age_binned'].cat.add_categories(['missing']).fillna('missing')
# datasets = datasets.drop('age', axis=1)

    # 標籤編碼
le = LabelEncoder()
le_columns = ['gender', 'device_type', 'ad_position', 'browsing_history', 'time_of_day']
for column in le_columns:
    datasets[column] = le.fit_transform(datasets[column])

    # 分割數據集
X = datasets.drop('click', axis=1)
y = datasets['click']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 標準化數據
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)  # 應該輸出 (樣本數量, 特徵數量)
print(X_test.shape)   # 應該輸出 (樣本數量, 特徵數量)
print('Data preprocessing done!')

model = Sequential()
model.add(Dense(16,activation='relu',input_dim=5))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=30,batch_size=32)
y_pred = model.predict(X_test)
y_pred= (y_pred > 0.5).astype(int)

acc = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)
confusion = confusion_matrix(y_test,y_pred)

print('Accuracy:',acc)
print('Classification report:',report)
print('Confusion matrix:',confusion)

# result = {}
# result['accuracy'] = acc
# result['classification_report'] = report
# result['confusion_matrix'] = confusion

# result = pd.DataFrame(result,index=[0])
# result.to_csv('result_ann.csv',index=False)

print('Model training done!')