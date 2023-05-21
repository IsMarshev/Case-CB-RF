from datetime import datetime
import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulations
import matplotlib.pyplot as plt  # plots
import seaborn as sns  # more plots
from dateutil.relativedelta import relativedelta  # working with dates with style
from scipy.optimize import minimize  # for function minimization
import statsmodels.formula.api as smf  # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from itertools import product  # some useful functions
from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings("ignore")


# функция отображения автокор. и част. автокор
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# Обработка данных
train = pd.read_csv("data/test.csv")
sub = pd.read_csv("data/predicted_test.csv")
train = pd.concat([train, sub])  # увеличиваем количество данных
train['Дата обращения'] = pd.to_datetime(train['Дата обращения']).dt.date

# Определение количества по  датам
count_end_ins = train[train['Тип обращения итоговый'] == 'Инцидент'].groupby('Дата обращения')[
    'Тип обращения итоговый'].count()
count_end_qwe = train[train['Тип обращения итоговый'] == 'Запрос'].groupby('Дата обращения')[
    'Тип обращения итоговый'].count()

# Период прогнозирования
time_for = pd.date_range(start='26/2/2018', end='31/3/2018')

# Для Инцидент, фактических
ads = pd.DataFrame(count_end_ins)
# tsplot(ads['Тип обращения итоговый'], lags=14)
# всё норм, говорить о цикличности/сеззоность некоректно, так как короткий срок
results = sm.tsa.statespace.SARIMAX(ads['Тип обращения итоговый'], order=(1, 0, 1),
                                    seasonal_order=(0, 1, 0, 31)).fit()
forecast = abs(results.forecast(steps=34))
# Построение графиков прогноза
plt.plot(time_for, forecast)
plt.xlabel('Дата')
plt.ylabel('Количество')
plt.title('Количество предсказанных Инцидентов (фактических)')
# plt.show()
# Поиск выбросов
n_f = pd.concat([forecast, ads['Тип обращения итоговый']])
Q1 = np.percentile(n_f, 25, method='midpoint')
Q3 = np.percentile(n_f, 75, method='midpoint')
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
upper_array = np.where(n_f >= upper)[0]
lower_array = np.where(n_f <= lower)[0]
upper_array1 = upper_array


# Для Запрос, фактических
ads = pd.DataFrame(count_end_qwe)
# tsplot(ads['Тип обращения итоговый'], lags=15)  # видна цикличность
# Избавляемся от цикличности
ads_diff = ads['Тип обращения итоговый'] - ads['Тип обращения итоговый'].shift(6)
ads_diff = ads_diff - ads_diff.shift(1)

results = sm.tsa.statespace.SARIMAX(ads['Тип обращения итоговый'], order=(7, 0, 2),
                                    seasonal_order=(0, 1, 0, 6)).fit()
forecast = abs(results.forecast(steps=34))

# Поиск выбросов
n_f = pd.concat([forecast, ads['Тип обращения итоговый']])
Q1 = np.percentile(n_f, 25, method='midpoint')
Q3 = np.percentile(n_f, 75, method='midpoint')
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
upper_array = np.where(n_f >= upper)[0]
lower_array = np.where(n_f <= lower)[0]
n_f.drop(index=upper_array, inplace=True)
n_f.drop(index=lower_array, inplace=True)
print('Аномалии по запросам не найдено')

print('Аномалии по инцидентам (даты):')
for i in upper_array1:
    if i>len(time_for):
        break
    print(time_for[i], end='\n')
