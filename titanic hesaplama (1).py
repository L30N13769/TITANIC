#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[45]:


df=pd.read_csv('titanic.csv')
df.head(10)


# In[46]:


missing_values = df.isnull().sum()
print(missing_values)


# In[47]:


print(df.dtypes)


# In[48]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Veri setinizi bir pandas DataFrame'e yükleyin
df = pd.read_csv('titanic.csv')

# 'Name' ve 'Ticket' özelliklerini sayısal bir formata dönüştürün
df['Name'] = pd.to_numeric(df['Name'], errors='coerce').fillna(0).astype(np.int64)
df['Ticket'] = pd.to_numeric(df['Ticket'], errors='coerce').fillna(0).astype(np.int64)

# 'Sex' özelliğini label encoding ile dönüştürün
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


# LabelEncoder'ı tanımlayın
le = LabelEncoder()

# 'Embarked' özelliğini label encoding ile dönüştürün
df['Embarked'] = le.fit_transform(df['Embarked'])



df.drop('Cabin', axis=1, inplace=True)

# Dönüştürülmüş veri setini yazdırın
print(df.head())



# In[49]:


print(df.dtypes)


# In[50]:


df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(np.int64)
df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce').fillna(0).astype(np.int64)

df['Age'].fillna(df['Age'].mean(), inplace=True)



# In[51]:


print(df.dtypes)


# In[52]:


missing_values = df.isnull().sum()
print(missing_values)


# In[53]:


df.head(10)


# In[54]:


print(df['Embarked'].value_counts())


# In[60]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Modeli eğit
model = LogisticRegression()  # Örnek olarak Logistic Regression modelini kullandım
model.fit(X_train, y_train)

# Tahmin olasılıklarını al
y_scores = model.predict_proba(X_test)

# ROC eğrisini hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])

# AUC'yi hesapla
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[61]:


from sklearn import tree
import matplotlib.pyplot as plt

# Modeli eğit
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Karar ağacını çiz
plt.figure(figsize=(15,10))
tree.plot_tree(model, filled=True)
plt.show()


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Özellikler ve hedef değişken
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Özellikler matrisi ve hedef vektör
X = df[features]
y = df[target]

# Veri setini eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri bir sözlükte sakla
models = {
    'linear_regression': LinearRegression(),
    'logistic_regression': LogisticRegression(),
    'ridge_regression': Ridge(),
    'lasso_regression': Lasso(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'svm': SVC(),
    'gradient_boosting': GradientBoostingClassifier(),
    'knn': KNeighborsClassifier()
}

from sklearn.metrics import accuracy_score, mean_squared_error

# Her model için
for name, model in models.items():
    # Modeli eğit
    model.fit(X_train, y_train)
    # Test seti üzerinde tahminler yap
    predictions = model.predict(X_test)
    
    # Modelin türüne göre uygun metriği seç ve skoru hesapla
    if name in ['linear_regression', 'ridge_regression', 'lasso_regression']:
        score = mean_squared_error(y_test, predictions)
        print(f'{name} modelinin mean squared error skoru: {score}')
    else:
        score = accuracy_score(y_test, predictions.round())  # Tahminleri en yakın tam sayıya yuvarla
        print(f'{name} modelinin accuracy skoru: {score}')


# In[58]:


from sklearn.preprocessing import StandardScaler

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Daha fazla iterasyon ve farklı bir solver ile modeli eğit
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Test seti üzerinde tahminler yap
predictions = model.predict(X_test_scaled)

# Tahminlerin doğruluğunu hesapla
score = accuracy_score(y_test, predictions)

# Modelin skorunu yazdır
print(f'Logistic Regression modelinin skoru: {score}')


# In[59]:


from sklearn.tree import DecisionTreeClassifier

# Modeli oluştur
model = DecisionTreeClassifier()

# Modeli eğit
model.fit(X_train, y_train)

# Test seti üzerinde tahminler yap
predictions = model.predict(X_test)

# Tahminlerin doğruluğunu hesapla
score = accuracy_score(y_test, predictions)

# Modelin skorunu yazdır
print(f'Decision Tree modelinin skoru: {score}')


# In[ ]:




