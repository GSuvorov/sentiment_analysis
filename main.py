import csv
import itertools
import random
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from scipy.sparse import spmatrix, coo_matrix
import sklearn.metrics as metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pymorphy2
from pyaspeller import Word

STOPWORDS = stopwords.words('russian')
DATA = 10000


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


def clean_tweets(a, pos_emoji, neg_emoji):

    a = ' '.join(a)
    a = ''.join(ch for ch, _ in itertools.groupby(a))

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
    result = result.lower()  # приведение к низкому регистру
    result = re.sub(r'\s+', ' ', result)  # лишние пробелы
    cleantweet = result.strip()
    cleantweet = ' '.join(word for word in cleantweet.split() if len(word) > 1)
    return cleantweet


def load_data():
    data = list()
    text = list()
    sentiment = list()
    pos_emoji = []
    neg_emoji = []

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
        cleanrow = clean_tweets(row, pos_emoji, neg_emoji)
        pos_txt_file.write(cleanrow + "\n")
        data.append([cleanrow, '1'])
    pos_txt_file.close()

    neg_csv_file = open('data/neg.csv', "rt", encoding='utf-8')
    reader = csv.reader(neg_csv_file)
    neg_txt_file = open("data/neg.txt", 'w')

    for row in reader:
        cleanrow = clean_tweets(row, pos_emoji, neg_emoji)
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


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Testing score")
    plt.legend(loc="best")
    return plt


def learning_curves(title, x, y, estimator):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8, random_state=0)
    plot_learning_curve(estimator, title, x, y, (0.4, 1.01), cv=cv, n_jobs=20)
    plt.show()


def main():
    token_pattern = r'\w+|[%s]' % string.punctuation
    vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=token_pattern, binary=False, stop_words=STOPWORDS,
                                 min_df=2)
    text, sentiment = load_data()
    X = vectorizer.fit_transform(text)
    print("Объем словаря: %s" % len(vectorizer.vocabulary_))
    classifier = SVC()
    print("Обучение модели")
    learning_curves("Learning Curves (SVM, RBF kernel", X, sentiment, classifier)



    # scorestrain, scorestest = crossvalidation(text, sentiment, vectorizer, classifier)
    # print("Отчет классификации - %s" % classifier)
    # print(metrics.classification_report(ytest, y_predicted))

    # datatrain=text[:10000]
    # datatest=text[10000:]
    # ytrain=sentiment[:10000]
    # ytest=sentiment[10000:]
    # classifier = SVC()
    # X_train = vectorizer.fit_transform(datatrain)
    # X_test=vectorizer.transform(datatest)
    # classifier.fit(X_train, ytrain)
    # y_predicted = classifier.predict(X_test)
    # print("Vocabulary Size: %s" % len(vectorizer.vocabulary_))
    # print('Точность: %s' % classifier.score(X_test, ytest))
    # print("Отчет классификации - %s" % classifier)
    # print(metrics.classification_report(ytest, y_predicted))
    # scorestrain, scorestest = crossvalidation(text, sentiment, vectorizer, classifier)


    # print(scorestrain)
    # print(scorestest)

    # classifier = NBSVM()
    # sentiment = list(map(int, sentiment))
    # sentiment =
    # .array(sentiment)
    # X_train=vectorizer.fit_transform(text[:160000])
    # X_test=vectorizer.transform(text[160000:])
    # y_train=sentiment[:160000]
    # y_test=sentiment[160000:]
    # classifier.fit(X_train, y_train)
    # y_predicted = classifier.predict(X_test)
    # print('Точность: %s' % classifier.score(X_test, y_test))
    # print("Отчет классификации - %s" % classifier)
    # print(metrics.classification_report(y_test, y_predicted))


if __name__ == '__main__':
    main()
