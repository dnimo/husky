# -*- coding:utf-8 -*-
import nltk, os, re
import pandas as pd
from gensim import corpora, models, similarities
from nltk.corpus import stopwords




def seg_depart(sentence):
    sentence_depart = nltk.word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    out_str = ''
    for word in sentence_depart:
        if word not in stop_words:
            out_str += word
            out_str += " "

    return out_str


if not os.path.exists('cut_training.txt'):
    out_filename = "cut_training.txt"
    inputs = pd.read_csv('training.csv')
    inputs = inputs['abstract']
    inputs = inputs.values.tolist()
    outputs = open(out_filename, 'w', encoding='utf-8')

    for line in inputs:
        line = re.sub(u'([^\u0041-\u005a\u0061-\u007a])', ' ', line)
        line_seg = seg_depart(line)
        outputs.write(line_seg + '\n')

    outputs.close()

# LDA part
fr = open('cut_training.txt', 'r', encoding='utf-8')

train = []
for line in fr:
    line = line.strip('\n')
    line = [word for word in line.split(' ')]
    train.append(line)

dictionay = corpora.Dictionary(train)

corpus = [dictionay.doc2bow(text) for text in train]

lda = models.LdaModel(corpus=corpus, id2word=dictionay, num_topics=5)


for topic in lda.print_topics(num_topics=5, num_words=5):
    print(topic)
# count = 0
# test_doc = train[:10]
# for i in test_doc:
#     count += 1
#     doc_bow = dictionay.doc2bow(i)
#     doc_lda = lda[doc_bow]
#     print('The ' + str(count) + ' topic:')
#     print(doc_lda)
#     for topic in doc_lda:
#         print("%s\tf\n"%(lda.print_topic(topic[0])))
