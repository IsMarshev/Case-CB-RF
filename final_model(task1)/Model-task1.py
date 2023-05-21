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
import joblib
import warnings
warnings.filterwarnings("ignore")

#после запуска решения создадутся 4 файла

class Solution():
    def __init__(self, path_train, path_test, model='catboost') -> None:
        '''
        path_train - путь до обучаюющего датасета
        path-тест - путь до тестового датасета
        model - тип модели(default: 'catboost', else:'lgbm')

        Модель написана для предсказаний на тех данных, которыми обладает 
        специалист на момент выставления текущего статуса,
        то есть в тренировочный датасет не входят временные признаки, 
        а также результат и статус(если бы модель использовала эти признаки, 
        то f1-macro был бы примерно 0.9-0.95, мы проверяли), остальные признаки были отобраны 
        различными экспериментами, в ходе которых подтвержадлись гипотезы о низком влиянии 
        признака на целевую переменную
        '''

        self.train = pd.read_csv(path_train)
        self.test = pd.read_csv(path_test)
        self.sub_copy = pd.read_csv(path_test)
        self.modelname = model
        self.clf = None
        self.tfidf = None

    def train_test_preprocessing(self):
        '''
        Предобраоботка датасетов для модели: 
        кодирование признаков, удаление NaN, составление 
        tfidf-матрица для столбцов Содержание
        '''
        train = self.train
        test = self.test
        print('started')
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
        test['Статус'] = test['Статус'].map({'Закрыт': 1, 'Отменен': 0})
        test['Критичность'] = test['Критичность'].map(
            {'4-Нет': 0, '3-Базовая': 1, '2-Повышенная': 2, '1-Особая': 3})
        test['Влияние'] = test['Влияние'].map(
            {'4-Нет влияния': 0, '3-Малое': 1, '2-Значительное': 2, '1-Всеохватывающее': 3})
        test['Тип обращения итоговый'] = test['Тип обращения итоговый'].map(
            {'Запрос': 0, 'Инцидент': 1})
        test['Тип обращения на момент подачи'] = test['Тип обращения на момент подачи'].map({'Запрос': 0, 'Инцидент': 1})
        print('preparing')
        drp = ['Сервис','Функциональная группа','Система','Место', 'Крайний срок',\
                'Дата обращения', 'Дата восстановления', 'Дата закрытия обращения',\
                'Тип переклассификации', 'Статус', 'Тип обращения на момент подачи']

        train.drop(drp,axis=1,inplace=True)
        test.drop(drp,axis=1,inplace=True)

        train.drop(train[train['Содержание'].isna()==True].index,inplace=True)
        train.reset_index(drop=True,inplace=True)

        test.drop(test[test['Содержание'].isna()==True].index,inplace=True)
        test.reset_index(drop=True,inplace=True)

        # nlp preparing part for columns 'Содержание'
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
        self.tfidf = vectorizer
        content_transformed = vectorizer.transform(test['Содержание'].dropna())
        content = pd.DataFrame.sparse.from_spmatrix(content_transformed)
        clms = [str(i) for i in range(content.shape[1])]
        content.columns = clms
        result = content
        self.test = pd.concat([result,test],axis=1)
        self.train.drop('Содержание',axis=1,inplace=True)
        self.train.drop('Решение             ',axis=1,inplace=True)
        self.test.drop('Содержание',axis=1,inplace=True)
        self.test.drop('Решение             ',axis=1,inplace=True)
        print('finished')

    def fitting(self):
        X = self.train.drop('Тип обращения итоговый',axis=1)
        y = self.train['Тип обращения итоговый']
        transformer = Normalizer().fit(X)
        X = transformer.transform(X)
        Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,random_state=42, test_size=0.2,stratify=y)
        if self.modelname == 'catboost':
            self.clf = CatBoostClassifier(depth = 8, l2_leaf_reg=4).fit(Xtrain, ytrain)
        elif self.modelname == 'lgbm':
            self.clf = LGBMClassifier().fit(Xtrain, ytrain)
        pred = self.clf.predict(Xtest)
        f1 = f1_score(ytest,pred,average='macro')
        print(f'Confusion matrix on val_data: \n{confusion_matrix(ytest, pred)}')
        print(f'F1-macro on val_data: {f1}')

    def submission(self):
        '''
        Формирование submission
        '''
        self.test.index = self.test.id
        indexes = self.test.index
        self.test.drop('id', axis=1, inplace=True)
        self.test.drop('Тип обращения итоговый', axis=1, inplace=True)
        transformer = Normalizer().fit(self.test)
        self.test = transformer.transform(self.test)
        predicts = self.clf.predict(self.test)
        self.sub_copy['Тип обращения итоговый'] = pd.Series(predicts)
        self.sub_copy['Тип переклассификации'] = 0 
        self.sub_copy['Тип обращения итоговый'] = self.sub_copy['Тип обращения итоговый'].map({0: 'Запрос', 1: 'Инцидент'})   

        for i in range(self.sub_copy.shape[0]):
            if self.sub_copy.at[i, 'Тип обращения на момент подачи'] == self.sub_copy.at[i, 'Тип обращения итоговый']:
                self.sub_copy.at[i, 'Тип переклассификации'] = 0
            elif (self.sub_copy.at[i, 'Тип обращения на момент подачи'] == 'Запрос') and (self.sub_copy.at[i, 'Тип обращения итоговый'] == 'Инцидент'):
                self.sub_copy.at[i, 'Тип переклассификации'] = 1
            elif (self.sub_copy.at[i, 'Тип обращения на момент подачи'] == 'Инцидент') and (self.sub_copy.at[i, 'Тип обращения итоговый'] == 'Запрос'):
                self.sub_copy.at[i, 'Тип переклассификации'] = 2
        self.sub_copy.to_csv('predicted_test.csv') #сохранение полного предикта для анализа вр.ряда
        final_submission = pd.DataFrame({'id': indexes, 'Тип переклассификации': self.sub_copy['Тип переклассификации'], 
                                         'Тип обращения итоговый': self.sub_copy['Тип обращения итоговый']})
        final_submission.to_csv("final_submission.csv") # настоящий главный сабмишн
    def export_models(self):
        '''
        Сохраняет предобученные модели для дальнейшей интеграции в наш прототип рекомендательной системы
        '''
        joblib.dump(self.clf, 'clf_model.pkl')
        joblib.dump(self.tfidf, 'tfidf_vectorizer.pkl')

sol = Solution('data/train.csv', 'data/test.csv', 'catboost')
sol.train_test_preprocessing() #подготовка датасета
sol.fitting() #обучение выбранной модели
sol.submission() # формирование полного трейн датасета
sol.export_models() # сохранение моделей дл интеграции