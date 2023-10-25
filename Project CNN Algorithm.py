import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay

from sklearn.metrics import   classification_report,accuracy_score,confusion_matrix

pd.options.mode.chained_assignment = None  # default='warn'


exoTrain = pd.read_csv("exoTrain.csv")
exoTest =  pd.read_csv("exoTest.csv")


def star_plot(x):
    
    sth = exoTrain.iloc[x, :]

    plt.figure(figsize=(30, 10))
    plt.plot(pd.Series([i for i in range(1, len(sth))]), sth[1:])

    plt.title('Distribution of flux values', fontsize=15)
    plt.xlabel('Flux values')
    plt.ylabel('Flux '+str(x))
    plt.legend(('Data'))    
    plt.show()
    
    
  
star_plot(0)
star_plot(2)
star_plot(37)
star_plot(39)


def star_hist(x):
    
    plt.figure(figsize=(16,8))
    plt.hist(exoTrain.iloc[x,:], bins=100)
    
    if exoTrain.iloc[x,0] == 2:
        plt.title("Gaussian Histogram of Exoplanets")
    else:
        plt.title("Gaussian Histogram of Non Exoplanets")
        
    plt.xlabel("Flux values")
    plt.ylabel('Flux '+str(x))
    plt.show()
    
    
star_hist(0)
star_hist(2)
star_hist(37)
star_hist(39)



def handle_outliers(dataset, num_iterations):
    
    #threshold = None
    dataset_handled = dataset

    for n in range(num_iterations):
        #for column in range(dataset_handled.shape[0]):
        for index, row in dataset_handled.iterrows():
            row_values = row.values
            row_maxidx, row_minidx = row_values.argmax(), row_values.argmin()
            row_mean = row_values.mean()

            #if np.abs(column_max/column_mean) >= threshold:
            dataset_handled.iloc[index][row_maxidx] = row_mean

            #if np.abs(column_min/column_mean) >= threshold:
            dataset_handled.iloc[index][row_minidx] = row_mean

    return dataset_handled

handled_dataset = handle_outliers(exoTrain.loc[:, exoTrain.columns != 'LABEL'], 2)


def star_plot_handled(x):
    
    sth = handled_dataset.iloc[x, :]

    plt.figure(figsize=(30, 10))
    plt.plot(pd.Series([i for i in range(1, len(sth))]), sth[1:])

    plt.title('Distribution of flux values', fontsize=15)
    plt.xlabel('Flux values')
    plt.ylabel('Flux '+str(x))
    plt.legend(('Data'))    
    plt.show()
    

star_plot_handled(0)
star_plot_handled(2)
star_plot_handled(37)
star_plot_handled(39)



def lable_change(y_train, y_test):
    labler = lambda x: 1 if x == 2 else 0
    y_train_01, y_test_01 = y_train.apply(labler), y_test.apply(labler)

    return y_train_01, y_test_01



def smote(x_train, y_train):

    over = SMOTE(sampling_strategy=0.2)
    under = RandomUnderSampler(sampling_strategy=0.3)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x_train_res, y_train_res = pipeline.fit_resample(x_train, y_train)

    return x_train_res, y_train_res



def smote2(x_test, y_test):

    over = SMOTE(sampling_strategy=0.2, k_neighbors = 4)
    under = RandomUnderSampler(sampling_strategy=0.3)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x_test_res, y_test_res = pipeline.fit_resample(x_test, y_test)

    return x_test_res, y_test_res



# Define training and testing datasets
def datasets():
    
    x_train, y_train = exoTrain.loc[:, exoTrain.columns != 'LABEL'], exoTrain.loc[:, 'LABEL']
    x_test, y_test = exoTest.loc[:, exoTest.columns != 'LABEL'], exoTest.loc[:, 'LABEL']
    
    x_train = handle_outliers(x_train, 2)
    x_train, y_train = smote(x_train, y_train)
    x_test, y_test = smote2(x_test, y_test)
  
    y_train, y_test = lable_change(y_train, y_test)
    
    n_features = x_train.shape[1]

    return x_train, y_train, x_test, y_test, n_features



# Graph train and test accuracy
def graph_acc(history):
    
    # Plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    # Plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()



# Confusion matrix
def conf_matrix(y_test, y_pred):

    matrix = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(matrix, columns=[0, 1], index = [0, 1])
    df_cm.index.name = 'Truth'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4) 
    sn.heatmap(df_cm, cmap="BuGn", annot=True, annot_kws={"size": 16})
    plt.show()
    
    return matrix



# Print prediction metrics
def prediction_metrics(y_test, y_pred, y_class_pred, matrix):
    
    FP = matrix[0][1] 
    FN = matrix[1][0]
    TP = matrix[1][1]
    TN = matrix[0][0]

    sens = TP/(TP+FN)
    spec = TN/(TN+FP) 
    g_mean = np.sqrt(sens * spec)

    accuracy = accuracy_score(y_test, y_class_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_class_pred)
    precision = precision_score(y_test, y_class_pred)
    recall = recall_score(y_test, y_class_pred)
    f1 = f1_score(y_test, y_class_pred)
    auc = roc_auc_score(y_test, y_pred)

    print('\n\t\t Prediction Metrics\n')
    print("Accuracy:\t", "{:0.3f}".format(accuracy))
    print("Precision:\t", "{:0.3f}".format(precision))
    print("Recall:\t\t", "{:0.3f}".format(recall))
    print("\nF1 Score:\t", "{:0.3f}".format(f1))
    print("ROC AUC:\t", "{:0.3f}".format(auc))
    print("Balanced\nAccuracy:\t", "{:0.3f}".format(balanced_accuracy))
    print("\nSensitivity:\t", "{:0.3f}".format(sens))
    print("Specificity:\t", "{:0.3f}".format(spec))
    print("Geometric Mean:\t", "{:0.3f}".format(g_mean))

    print ("\nClassification report :\n",(classification_report(y_test,precision)))

def cnn_model():

    # Data preparation
    x_train, y_train, x_test, y_test, n_features = datasets()
    x_train, y_train = shuffle(x_train, y_train) # shuffle the data to avoid stagnant 0.0000e+00 val_accuracy

    # Architecture
    model = Sequential()
    model.add(Reshape((3197, 1), input_shape=(3197,)))
    model.add(Conv1D(filters=10, kernel_size=2, activation='relu', input_shape=(n_features, 1), kernel_regularizer='l2'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(48, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(18, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Representation of architecture
    print(model.summary())

    # Compile model
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.94)

    model.compile(optimizer = Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy'])

    # Fit model
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(x_train, y_train, validation_split = 0.2, batch_size=64, callbacks=[early_stop], epochs=30, verbose=2)

    # Evaluate the model
    _, train_acc = model.evaluate(x_train, y_train, verbose=2)
    _, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

    # Prediction
    y_class_pred = (model.predict(x_test) > 0.5).astype("int32")
    y_pred = model.predict(x_test)

    # Accuracy graph
    graph_acc(history)

    # Confustion matrix
    matrix = conf_matrix(y_test, y_class_pred)

    # Metrics
    prediction_metrics(y_test, y_pred, y_class_pred, matrix)


cnn_model()




















































































