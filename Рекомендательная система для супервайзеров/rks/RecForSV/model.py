import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

class Solution():
    def __init__(self, test_ds, model='catboost') -> None:
        '''
        path_train - путь до обучаюющего датасета
        path-тест - путь до тестового датасета
        model - тип модели(default: 'catboost', else:'lgbm')
        '''

        self.train = pd.read_csv('static/data/train.csv')
        self.test = test_ds
        self.modelname = model
        self.clf = None

    def train_test_preprocessing(self):
        '''
        Предобраоботка датасетов для модели: 
        кодирование признаков, удаление NaN, составление 
        tfidf-матрицы для столбцов Содержание и Результат
        '''
        train = self.train
        test = self.test
        print('good')
        train['Приоритет'] = train['Приоритет'].map(
            {'0-Критический': 3, '1-Высокий': 2, '2-Средний': 1, '3-Низкий': 0})
        train['Статус'] = train['Статус'].map({'Закрыт': 1, 'Отменен': 0})
        train['Критичность'] = train['Критичность'].map(
            {'4-Нет': 0, '3-Базовая': 1, '2-Повышенная': 2, '1-Особая': 3})
        train['Влияние'] = train['Влияние'].map(
            {'4-Нет влияния': 0, '3-Малое': 1, '2-Значительное': 2, '1-Всеохватывающее': 3})
        train['Тип обращения итоговый'] = train['Тип обращения итоговый'].map(
            {'Запрос': 0, 'Инцидент': 1})
        train['Тип обращения на момент подачи'] = train['Тип обращения на момент подачи'].map({
                                                                                              'Запрос': 0, 'Инцидент': 1})
        test['Приоритет'] = test['Приоритет'].map(
            {'0-Критический': 3, '1-Высокий': 2, '2-Средний': 1, '3-Низкий': 0})
        # test['Статус'] = test['Статус'].map({'Закрыт': 1, 'Отменен': 0})
        test['Критичность'] = test['Критичность'].map(
            {'4-Нет': 0, '3-Базовая': 1, '2-Повышенная': 2, '1-Особая': 3})
        test['Влияние'] = test['Влияние'].map(
            {'4-Нет влияния': 0, '3-Малое': 1, '2-Значительное': 2, '1-Всеохватывающее': 3})
        print('preparing')
        drp = ['Сервис','Функциональная группа','Система','Место', 'Крайний срок',\
                'Дата обращения', 'Дата восстановления', 'Дата закрытия обращения',\
                'Тип переклассификации','Тип обращения на момент подачи','Статус']

        train.drop(drp,axis=1,inplace=True)
        # test.drop(drp,axis=1,inplace=True)

        train.drop(train[train['Содержание'].isna()==True].index,inplace=True)
        train.reset_index(drop=True,inplace=True)

        # nlp preparing part for columns 'Содержание' и 'Решение'
        def preprocess_text(text):
            text = re.sub(r"[^а-яА-ЯёЁa-zA-Z0-9]", " ", text)
            text = re.sub(r"\s+", " ", text)
            tokens = word_tokenize(text.lower(), language='russian')
            stop_words = set(stopwords.words("russian"))
            stop_words.remove('не')
            tokens = [word for word in tokens if word not in stop_words]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)
        
        train['Содержание'] = train['Содержание'].apply(preprocess_text)
        train['Решение             '] = train['Решение             '].apply(preprocess_text)
        test['Содержание'] = test['Содержание'].apply(preprocess_text)
        print('preparing')
        #tf-idf
        vectorizer = TfidfVectorizer()
        content_transformed = vectorizer.fit_transform(train['Содержание'].dropna())
        content = pd.DataFrame.sparse.from_spmatrix(content_transformed)
        clms = [str(i) for i in range(content.shape[1])]
        content.columns=clms
        result = content
        self.train = pd.concat([result,train],axis=1)

        content_transformed = vectorizer.transform(test['Содержание'].dropna())
        content = pd.DataFrame.sparse.from_spmatrix(content_transformed)
        clms = [str(i) for i in range(content.shape[1])]
        content.columns = clms
        result = content
        self.test = pd.concat([result,test],axis=1)
        self.train.drop('Содержание',axis=1,inplace=True)
        self.train.drop('Решение             ',axis=1,inplace=True)
        self.test.drop('Содержание',axis=1,inplace=True)
        print('good')

    def fitting(self):
        X = self.train.drop('Тип обращения итоговый',axis=1)
        y = self.train['Тип обращения итоговый']
        transformer = Normalizer().fit(X)
        X = transformer.transform(X)
        Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=42, test_size=0.2,stratify=y)
        if self.modelname == 'catboost':
            self.clf = CatBoostClassifier().fit(Xtrain, ytrain)
        elif self.modelname == 'lgbm':
            self.clf = LGBMClassifier().fit(Xtrain, ytrain)
        pred = self.clf.predict(Xtest)
        f1 = f1_score(ytest,pred,average='macro')

    def submission(self):
        predicts = self.clf.predict_proba(self.test)
        return predicts

