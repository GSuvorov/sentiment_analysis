import random
import csv
import itertools
import re
import pymorphy2
from pyaspeller import Word

DATA = 10000

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
                word = ' обсценнаялексика '
            else:
                word = word
        result_string.append(word)
    return " ".join(result_string)

def clean_tweets(a, pos_emoji, neg_emoji, obscene, pos_words, neg_words):
    a = ' '.join(a)

    for p in list(pos_emoji):
          a = a.replace(p, ' положительныйэмотикон ')
    for p in list(neg_emoji):
          a = a.replace(p, ' негативныйэмотикон ')

    for p in list(pos_words):
          a = a.replace(p, ' положительноеслово ')
    for p in list(neg_words):
          a = a.replace(p, ' негативноеслово ')

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
    data,text,sentiment = list(),list(),list()
    pos_emoji,neg_emoji,obscene,pos_words,neg_words = [],[],[],[],[]

    file = open('../data/pos_words.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        pos_words.append(line)

    file = open('../data/neg_words.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        neg_words.append(line)

    file = open('../data/obscene.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        obscene.append(line)

    file = open('../data/possmile.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        pos_emoji.append(line)

    file = open('../data/negsmile.txt', "rt", encoding='utf-8')
    for line in file:
        line = line.replace("\n", "")
        neg_emoji.append(line)

    pos_csv_file = open('../data/pos.csv', "rt", encoding='utf-8')
    reader = csv.reader(pos_csv_file)

    pos_txt_file = open('../data/pos.txt', 'w')
    for row in reader:
        cleanrow = clean_tweets(row, pos_emoji, neg_emoji, obscene, pos_words, neg_words)
        pos_txt_file.write(cleanrow + "\n")
        data.append([cleanrow, '1'])
    pos_txt_file.close()

    neg_csv_file = open('../data/neg.csv', "rt", encoding='utf-8')
    reader = csv.reader(neg_csv_file)
    neg_txt_file = open('../data/neg.txt', 'w')

    for row in reader:
        cleanrow = clean_tweets(row, pos_emoji, neg_emoji, obscene, neg_words, neg_words)
        neg_txt_file.write(cleanrow + "\n")
        data.append([cleanrow, '0'])
    neg_txt_file.close()

    random.shuffle(data)
    data = data[:DATA]
    for i in data:
        text.append(i[0])
        sentiment.append(i[1])

    return text, sentiment