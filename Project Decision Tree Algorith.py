


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from sklearn.metrics import   classification_report,accuracy_score,confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split 


train_data = pd.read_csv("exoTrain.csv")
test_data =  pd.read_csv("exoTest.csv")


def star_plot(x):
    
    sth = train_data.iloc[x, :]

    plt.figure(figsize=(30, 10))
    plt.plot(pd.Series([i for i in range(1, len(sth))]), sth[1:])

    plt.title('Distribution of flux values', fontsize=15)
    plt.xlabel('Flux values')
    plt.ylabel('Flux intensity')
    plt.legend(('Data {}'))    
    plt.show()
    
    
  
#"""  
star_plot(0)
star_plot(2)
star_plot(37)
star_plot(39)
#"""


def star_hist(x):
    
    plt.figure(figsize=(16,8))
    plt.hist(train_data.iloc[x,:], bins=100)
    
    if train_data.iloc[x,0] == 2:
        plt.title("Gaussian Histogram of Exoplanets")
    else:
        plt.title("Gaussian Histogram of Non Exoplanets")
        
    plt.xlabel("Flux values")
    plt.show()


star_hist(0)
star_hist(2)
star_hist(37)
star_hist(39)



from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


model = SMOTE()
ov_train_x,ov_train_y = model.fit_resample(train_data.drop('LABEL',axis=1), train_data['LABEL'])
ov_train_y = ov_train_y.astype('int')
ov_train_y.value_counts().reset_index().plot(kind='bar', x='index', y='LABEL')


x_train, x_test, y_train, y_test = train_test_split(ov_train_x, ov_train_y, test_size=0.1, random_state=42)



#Decision Tree
from sklearn.tree import DecisionTreeClassifier

ds_model = DecisionTreeClassifier(max_depth=5, random_state=13)
ds_model.fit(x_train,y_train)

prediction=ds_model.predict(x_test)


print('\nValidation accuracy of Decision Tree is', accuracy_score(prediction,y_test))
print ("\nClassification report :\n",(classification_report(y_test,prediction)))


#Confusion matrix
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,cmap="viridis",fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)




