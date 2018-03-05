#Возьмите готовую модель word2vec (см. последнее практическое занятие).
#Вычислите топ 10 слов для слова "начальник".
#Определите топ 5 близких к слову "начальник" по русскому
#ворд-нету: http://ruwordnet.ru/en/
#или по РуТез http://www.labinform.ru/pub/ruthes/
#(любым способом, опишите, как Вы искали близкие слова).
#Каков процент пересечения? Какие, полученные в выбранной Вами модели слова,
#на Ваш взгляд, попали в топ 10 близких ошибочно.
#Попробуйте прокомментировать, почему

#слова искала руками через поиск на сайте
rus_wordnet=['ШЕФ', 'БОСС', 'РУКОВОДИТЕЛЬ', 'ГЛАВА', 'НАЧАЛЬНИЦА']

import sys
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

f = 'text.txt'
data = gensim.models.word2vec.LineSentence(f)

model = gensim.models.Word2Vec(data, size=500, window=10, min_count=2, sg=0)

model.init_sims(replace=True)
print(len(model.vocab))
model.save('my.model')

m = 'ruscorpora_1_300_10.bin.gz'
if m.endswith('.vec.gz'):
    model = gensim.models.Word2Vec.load_word2vec_format(m, binary=False)
elif m.endswith('.bin.gz'):
    model = gensim.models.Word2Vec.load_word2vec_format(m, binary=True)
else:
    model = gensim.models.Word2Vec.load(m)

model.init_sims(replace=True)

words = ['начальник_NOUN']

for word in words:
    # есть ли слово в модели? Может быть, и нет
    if word in model:
        print(word)
        # смотрим на вектор слова (его размерность 300, смотрим на первые 10 чисел)
        print(model[word][:10])
        # выдаем 10 ближайших соседей слова:
        for i in model.most_similar(positive=[word], topn=10):
            # слово + коэффициент косинусной близости
            print(i[0], i[1])
        print('\n')
    else:
        # Увы!
        print(word + ' is not present in the model')
