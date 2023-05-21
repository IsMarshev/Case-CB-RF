from django import forms

class Input(forms.Form):
    IMPACT_CHOICES = (
        ('4-Нет влияния','4-Нет влияния'),
        ('3-Малое','3-Малое'),
        ('2-Значительное','2-Значительное'),
        ('1-Всеохватывающее','1-Всеохватывающее')
    )
    SEVERITY_CHOICES = (
        ('4-Нет','4-Нет'),
        ('3-Базовая','3-Базовая'),
        ('2-Повышенная','2-Повышенная'),
        ('1-Особая','1-Особая')
    )
    PRIORITY_CHOICES = (
        ('3-Низкий', '3-Низкий'),
        ('2-Средний','2-Средний'),
        ('1-Высокий','1-Высокий'),
        ('0-Критический','0-Критический')
    )
    content = forms.CharField(label='Содержание', widget=forms.Textarea)
    impact = forms.ChoiceField(label='Влияние', choices=IMPACT_CHOICES)
    severity = forms.ChoiceField(label='Критичность', choices=SEVERITY_CHOICES)
    priority = forms.ChoiceField(label='Приоритет', choices=PRIORITY_CHOICES)
