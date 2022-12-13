import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, plot_confusion_matrix, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

#--------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------------------------------------------------#

def hplotConfusionMatrix(yTrue, yPred):
    cm = confusion_matrix(yTrue, yPred)
    sns.heatmap(cm)

def graphData(df):

    sns.pairplot(
    df,
    x_vars=["fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "residual.sugar", "free.sulfur.dioxide", "free.sulfur.dioxide", "density", "density", "density", "density", "density"],
    y_vars=["quality"],
    )

    #plt.show()

    sns.distplot(df['quality'])

    #plt.show()

    sns.scatterplot(x='density', y='alcohol', data=df)

    #plt.show()


def main():

    # get data
    df = pd.read_csv("winequalityReds.csv")
    
    # exmaine the data
    graphData(df)

    # split the data
    x = df.drop('quality',axis=1)
    y = df['quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    classifiers = [LogisticRegression(solver='saga', l1_ratio=1, penalty='elasticnet', max_iter = 100000),
                    Perceptron(max_iter = 1000000),
                    SGDClassifier(max_iter = 1000000),
                    RidgeClassifier(max_iter = 1000000)]

    # uncomment these two lines out to use MinMaxScaler
    # min_max_scaler = MinMaxScaler()
    # x_train = min_max_scaler.fit_transform(x_train)

    # uncomment these two lines out to use MaxAbsScaler
    abs_scaler = MaxAbsScaler()
    x_train = abs_scaler.fit_transform(x_train)

    # uncomment these two lines out to use RobustScaler
    # rbt_scaler = RobustScaler()
    # x_train = rbt_scaler.fit_transform(x_train)

    # train the model

    for classifier in classifiers:


        classifier.fit(x_train,y_train)

        print(classifier.intercept_)

        pred_train = classifier.predict(x_train)

        pred_test = classifier.predict(x_test) # predictions

        print('MAE:', metrics.mean_absolute_error(y_train, pred_train))
        print('MSE:', metrics.mean_squared_error(y_train, pred_train))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, pred_train)))

        print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
        print('MSE:', metrics.mean_squared_error(y_test, pred_test))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))

        correct = 0
        incorrect = 0
        for pred, gt in zip(pred_test, y_test):
            if pred == gt: correct += 1
            else: incorrect += 1
        print(f"# Correct: {correct}, # Incorrect: {incorrect}, Accuracy: {correct/(correct + incorrect): 5.2}")

        # plot the results
        fig = plt.figure()
        sns.scatterplot(x=y_train, y=pred_train)

        if classifier != LinearRegression():
            plot_confusion_matrix(classifier, x_test, y_test)

            fig = plt.figure()
            hplotConfusionMatrix(y_test, pred_test)
            plt.show()


    # try linear regression too
    lm = LinearRegression()
    lm.fit(x_train, y_train)

    print(lm.intercept_)

    pred_train = lm.predict(x_train)
    pred_test = lm.predict(x_test) # predictions

    print('MAE:', metrics.mean_absolute_error(y_train, pred_train))
    print('MSE:', metrics.mean_squared_error(y_train, pred_train))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, pred_train)))

    print('MAE:', metrics.mean_absolute_error(y_test, pred_test))
    print('MSE:', metrics.mean_squared_error(y_test, pred_test))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_test)))

    # plot the results
    fig = plt.figure()
    sns.scatterplot(x=y_train, y=pred_train)
    plt.show()


main()