import pandas as pd

words = pd.read_csv('nagisa_stopwords.csv')

lt = words['words']

print(list(lt))
