from django.shortcuts import render
from .form import Input
from .model import *
import pandas as pd
def main(request):
    accident = 0
    req = 0
    if request.method == 'POST':
        form = Input(request.POST)
        if form.is_valid():
            # Обработка валидных данных формы
            # Доступ к значениям полей:
            content = form.cleaned_data['content']
            impact = form.cleaned_data['impact']
            severity = form.cleaned_data['severity']
            priority = form.cleaned_data['priority']
            test = pd.DataFrame({'Содержание': [content],'Влияние': [impact],'Приоритет': [priority],'Критичность': [severity]})
            sol = Solution(test_ds=test, model='lgbm')
            sol.train_test_preprocessing()
            sol.fitting()
            predict=sol.submission()
            print(predict)
            accident =  round(predict[0][1]*100)
            req = round(predict[0][0]*100)
            print(req,accident)
    else:
        form = Input()
    return render(request,'main.html', {'form': form,'accident':accident,'request':req})