import re
import csv
import itertools
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import spmatrix, coo_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

STOPWORDS = stopwords.words('russian')
DATA = 2000


class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    def __init__(self, alpha=1, C=1, beta=0.25, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        coef_, intercept_ = self.nbsvm(X, y)
        self.coef_ = coef_
        self.intercept_ = intercept_
        return self

    def nbsvm(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p / np.abs(p).sum()) - np.log(q / np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix((r, (indices, indices)), shape=(len(r), len(r)))
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(C=self.C, fit_intercept=self.fit_intercept, max_iter=10000).fit(X_scaled, y)
        mean_mag = np.abs(lsvc.coef_).mean()
        coef_ = (1 - self.beta) * mean_mag * r + self.beta * (r * lsvc.coef_)
        intercept_ = (1 - self.beta) * mean_mag * b + self.beta * lsvc.intercept_

        return coef_, intercept_


def clean_tweets(a):
    a = ' '.join(a)
    punctuation = ['.', ',', '-']
    pos_smile = [':)', ';)', ':D', ';D', ')', ':*', ':3', '^^', '^_^', '^-^', '*_*', '*', '^_~', '*-*', '♥', '❤', ':-)']
    neg_smile = [':(', ':|', '._.', ';(', ':~(', '=(', ':-(', '(', ':(', 'D:']

    # pos_smile=[':)',':D',';)',':-)',':P','=)','(:',';-)','=D','=]','D:',';D',':]','']
    # neg_smile=[':(','=(',';(',':-(','=/','=(']

    # for p in list(pos_smile):
    #    a = a.replace(p, ' [положительныйэмотикон] ')
    # for p in list(neg_smile):
    #    a = a.replace(p, ' [негативныйэмотикон] ')

    # result = re.sub(r"[хxХX]+[DdДд]+\s+", ' положительныйэмотикон ', a)  # xD
    # result = re.sub(r"[:;)]+\s+", ' положительныйэмотикон ', result)  # :) ;)
    # result = re.sub(r"[:;(]+\s+", ' положительныйэмотикон ', result)  # :( ;(
    # result = re.sub(r"[oO0оО]_+[oO0оО]", ' негативныйэмотикон ', result)  # O_o

    result = re.sub(r'(?:@[\w_]+)', '', a)  # упоминания
    result = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", '', result)  # хештеги
    result = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', result)  # ссылки
    result = re.sub(r'RT', '', result)  # RT
    result = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', ' ', result)  # цифры
    result = re.sub(r'[^а-яеёА-ЯЕЁ0-9-_*.]', ' ', result)  # символы
    result = ''.join(ch for ch, _ in itertools.groupby(result))  # повторяющиеся буквы
    result = re.sub(r'[a-zA-Z]+', ' ', result)

    result = result.lower()
    # for p in list(punctuation):
    #    result = result.replace(p, ' ')
    result = re.sub(r'\s+', ' ', result)
    cleantweet = result.strip()
    cleantweet = ' '.join(word for word in cleantweet.split() if len(word) > 2)
    return cleantweet


def load_data():
    data = list()
    text = list()
    sentiment = list()

    file = open('data/pos.csv', "rt", encoding='utf-8')
    reader = csv.reader(file)
    f = open("data/data.txt", 'w')
    for row in reader:
        cleanrow = clean_tweets(row)
        f.write(cleanrow + "\n")
        data.append([cleanrow, '1'])
    f.close()

    file = open('data/neg.csv', "rt", encoding='utf-8')
    reader = csv.reader(file)
    f = open("data/data.txt", 'a')
    for row in reader:
        cleanrow = clean_tweets(row)
        f.write(cleanrow + "\n")
        data.append([cleanrow, '-1'])
    f.close()

    random.shuffle(data)
    data = data[:DATA]
    for i in data:
        text.append(i[0])
        sentiment.append(i[1])
    return text, sentiment


def crossvalidation(x, y, vectorizer, classifier):
    X_folds = np.array_split(x, 10)
    y_folds = np.array_split(y, 10)
    scorestrain = list()
    scorestest = list()
    for k in range(10):
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def learning_curves(title, x, y, estimator):
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(estimator, title, x, y, (0.1, 1.), cv=cv, n_jobs=4)
    plt.show()


def main():
    token_pattern = r'\w+|[%s]' % string.punctuation
    vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=token_pattern, binary=False)
    text, sentiment = load_data()
    X = vectorizer.fit_transform(text)
    classifier = LinearSVC()
    scorestrain, scorestest = crossvalidation(text, sentiment, vectorizer, classifier)
    # classifier=SVC(gamma=0.001)
    # learning_curves("Learning Curves (SVM, Linear kernel, $\gamma=0.001$",X,sentiment,classifier)


if __name__ == '__main__':
    main()
