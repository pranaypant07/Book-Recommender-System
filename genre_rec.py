import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mongodb_connection import get_genre_collection

bisac = pd.read_csv('H:/consolidate/input_files/bisac_input.csv')


bisac[bisac['Bisac'].isnull()] = 'None'
bisacs = []
isbns = []
for i in range(0, len(bisac)):
    line = bisac.iloc[i, 1]
    isbns.append(str(bisac.iloc[i, 0]).lower())
    bisacl = line.split(';')
    p = ''
    for j in bisacl:
        first = j.split('/')[0].replace(' ', '').replace(',', '~')
        p = p + ' ' + first
        p = p.strip()
    bisacs.append(p)


biv = CountVectorizer(max_df=0.85)
bisac_count_vector = biv.fit_transform(bisacs)
bisacsmatrix = biv.transform(bisacs)

bisacsim = cosine_similarity(bisacsmatrix, bisacsmatrix)

def get_top_books_collection():
    s = np.argsort(bisacsim, axis=1)
    top_books_collection = {}
    for i in range(len(isbns)):
        top_100_books_idx = s[i][-100:]
        top_100_books = []
        for j in top_100_books_idx:
            top_100_books.append(str(isbns[j]))
        top_books_collection[str(isbns[i])] = top_100_books
    collection = get_genre_collection()
    collection.insert(top_books_collection)

if __name__ == '__main__':
    get_top_books_collection()
