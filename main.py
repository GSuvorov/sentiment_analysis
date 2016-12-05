from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import spmatrix, coo_matrix
from sklearn.base import BaseEstimator
import sklearn.metrics as metrics
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from pyaspeller import Word
import pandas as pd
import numpy as np
import pymorphy2
import itertools
import random
import string
import csv
import re

STOPWORDS = stopwords.words('russian')
DATA = 5000


class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    def __init__(self, alpha=1, C=1, beta=0.25, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[self._fit_binary(X, y == class_) for class_ in self.classes_])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p / np.abs(p).sum()) - np.log(q / np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(C=self.C, fit_intercept=self.fit_intercept, max_iter=10000).fit(X_scaled, y)

        mean_mag = np.abs(lsvc.coef_).mean()

        coef_ = (1 - self.beta) * mean_mag * r + self.beta * (r * lsvc.coef_)

        intercept_ = (1 - self.beta) * mean_mag * b + self.beta * lsvc.intercept_

        return coef_, intercept_


def spellchecking(cleantweet):
    list_word = cleantweet.split()
    result_string = []
    morph = pymorphy2.MorphAnalyzer()

    for word in list_word:
        check = Word(word)
        if not check.correct:
            if (len(check.variants)) == 0:
                pass
            else:
                print(word)
                print(check.variants)
                word = check.variants[0]
        else:
            pass

        word = morph.parse(word)[0].normal_form
        word = str(word)
        result_string.append(word)
    return " ".join(result_string)


def normalizewords(cleantweet):
    list_word = cleantweet.split()
    result_string = []
    morph = pymorphy2.MorphAnalyzer()

    for word in list_word:
        word = morph.parse(word)[0].normal_form
        word = str(word)
        result_string.append(word)
    return " ".join(result_string)


def obscene_check(a, obscene):
    list_word = a.split()
    result_string = []
    for word in list_word:
        for p in list(obscene):
            if word == p:
                word = ' обсценная лексика '
            else:
                word = word
        result_string.append(word)
    return " ".join(result_string)


def clean_tweets(a, pos_emoji, neg_emoji, obscene):
    a = ' '.join(a)

    for p in list(pos_emoji):
        a = a.replace(p, ' положительныйэмотикон ')
    for p in list(neg_emoji):
        a = a.replace(p, ' негативныйэмотикон ')

    result = re.sub(r'(?:@[\w_]+)', '', a)  # упоминания
    result = re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', result)  # хештеги
    result = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', result)  # ссылки
    result = re.sub(r'RT', '', result)  # RT
    result = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)', ' ', result)  # цифры
    result = re.sub(r'[^а-яеёА-ЯЕЁ0-9-_*.]', ' ', result)  # символы
    result = re.sub(r'[a-zA-Z.,?!@#$%^&*()_+-]+', ' ', result)  # англ слова и символы
    result = ''.join(ch for ch, _ in itertools.groupby(result))  # повторяющиеся буквы
    result = result.lower()  # приведение к низкому регистру

    result = re.sub(r'\s+', ' ', result)  # лишние пробелы

    cleantweet = ' '.join(word for word in result.split() if len(word) > 2)  # удаление слов длинной 1,2 символа

    cleantweet = obscene_check(cleantweet, obscene)  # проверка на наличие обсценной лексики

    cleantweet = cleantweet.strip()

    return cleantweet


def load_data():
    data = list()
    text = list()
    sentiment = list()
    pos_emoji = []
    neg_emoji = []
    obscene = []

    file = open('data/obscene.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        obscene.append(line)

    file = open('data/possmile.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        pos_emoji.append(line)

    file = open('data/negsmile.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        neg_emoji.append(line)

    pos_csv_file = open('data/pos.csv', "rt", encoding='utf-8')
    reader = csv.reader(pos_csv_file)
    pos_txt_file = open("data/pos.txt", 'w')
    for row in reader:
        cleanrow = clean_tweets(row, pos_emoji, neg_emoji, obscene)
        pos_txt_file.write(cleanrow + "\n")
        data.append([cleanrow, '1'])
    pos_txt_file.close()

    neg_csv_file = open('data/neg.csv', "rt", encoding='utf-8')
    reader = csv.reader(neg_csv_file)
    neg_txt_file = open("data/neg.txt", 'w')

    for row in reader:
        cleanrow = clean_tweets(row, pos_emoji, neg_emoji, obscene)
        neg_txt_file.write(cleanrow + "\n")
        data.append([cleanrow, '0'])
    neg_txt_file.close()

    random.shuffle(data)
    data = data[:DATA]
    for i in data:
        text.append(i[0])
        sentiment.append(i[1])

    return text, sentiment


def crossvalidation(x, y, vectorizer, classifier):
    X_folds = np.array_split(x, 5)
    y_folds = np.array_split(y, 5)
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.2, 1.0, 5)):
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
    print('train scores')
    print(train_scores)
    print('test scores')
    print(train_scores)
    return plt


def learning_curves(title, x, y, estimator):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=0)
    plot_learning_curve(estimator, title, x, y, (0.4, 1.02), cv=cv, n_jobs=-1)
    plt.show()


def main():
    token_pattern = r'\w+|[%s]' % string.punctuation
    vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=token_pattern, binary=False, stop_words=STOPWORDS,
                                 min_df=2)
    text, sentiment = load_data()
    X = vectorizer.fit_transform(text)
    print("Объем словаря: %s" % len(vectorizer.vocabulary_))
    # classifier = SVC()
    # classifier = MultinomialNB()
    classifier = RandomForestClassifier(n_estimators=10, n_jobs=-1)
    print("Обучение модели")

    word_freq_df = pd.DataFrame(
        {'term': vectorizer.get_feature_names(), 'occurrences': np.asarray(X.sum(axis=0)).ravel().tolist()})
    word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])
    print(word_freq_df.sort_values('occurrences', ascending=False))

    learning_curves("Learning Curves" + str(classifier), X, sentiment, classifier)



    # X_train, X_test, y_train, y_test = train_test_split(X, sentiment, test_size = 0.8, random_state = 42)
    # classifier.fit(X_train,y_train)
    # prediction = classifier.predict(X_test)
    # print(metrics.classification_report(y_test, prediction))


if __name__ == '__main__':
    main()
