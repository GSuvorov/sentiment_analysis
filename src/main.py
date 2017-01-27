# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import numpy as np
import string
import load_data

STOPWORDS = stopwords.words('russian')

def crossvalidation(x, y, vectorizer, classifier):
    X_folds = np.array_split(x, 10)
    y_folds = np.array_split(y, 10)
    scorestrain = list()
    scorestest = list()
    for k in range(5):
        print(k)
        X_train = list(X_folds)
        X_test = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test = y_train.pop(k)
        y_train = np.concatenate(y_train)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        scorestrain.append(classifier.fit(X_train, y_train).score(X_train, y_train))
        scorestest.append(classifier.fit(X_train, y_train).score(X_test, y_test))
        y_predicted = classifier.predict(X_test)
        print("*****")
        print("Отчет классификации - %s" % classifier)
        print(metrics.classification_report(y_test, y_predicted))
    return scorestrain, scorestest

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1, 10)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    font = {'family': 'Droid Sans',
            'weight': 'normal',
            'size': 14}
    plt.rc('font', **font)
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(u"Объем тренировочной выборки")
    plt.ylabel(u"Точность")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label=u"Точность обучения")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label=u"Точность тестирования")

    plt.legend(loc="best")
    print('train scores')
    print(train_scores)
    print('test scores')
    print(train_scores)
    return plt

def learning_curves(title, x, y, estimator):
    cv = ShuffleSplit(n_splits=10, test_size=0.2)
    plot_learning_curve(estimator, title, x, y, (0.4, 1.02), cv=cv, n_jobs=-1)
    plt.show()

def term_freq(vectorizer):
    word_freq_df = pd.DataFrame(
        {'term': vectorizer.get_feature_names(), 'occurrences': np.asarray(X.sum(axis=0)).ravel().tolist()})
    word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])
    word_freq_df.sort_values('occurrences', ascending=False).to_csv('data/term_frequences.csv')


def main():

    token_pattern = r'\w+|[%s]' % string.punctuation
    vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=token_pattern, binary=False,stop_words=STOPWORDS,min_df=2)
    text, sentiment = load_data.load_data()
    X = vectorizer.fit_transform(text)
    print("Объем словаря: %s" % len(vectorizer.vocabulary_))
    classifier = SVC()
    print("Обучение модели")
    term_freq(vectorizer)
    learning_curves(u'Кривая обучения', X, sentiment, classifier)

    # X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size = 0.2)
    # gamma_range = 10.0 ** np.arange(-4, 4)
    # param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
    # grid = GridSearchCV(classifier, param_grid)
    # grid.fit(X_train, y_train)
    # print("The best classifier is: ", grid.best_estimator_)
    # print(grid.best_score_)
    # classifier.fit(X_train,y_train)
    # prediction = classifier.predict(X_test)
    # print(metrics.classification_report(y_test, prediction))


if __name__ == '__main__':
    main()
