# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 05:02:19 2018

@author: sharmas
"""
# This is a code which looks at the books in an online library and looks at their individual aspects by importing the relevant information from a SQL server into txt files for each feature
# and using the same generates the similarity between all the books in the library to come up with the content based aspect of a recommender system.
# Later on we mix this code with the checkouts information to run a hybrid recommender system.

#####-----SEGMENT 1-----#####
# importing all modules we need for this task

import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
import re
import time
from mongodb_connection import get_user_collection, get_library_collection

start_time = time.time()



# creating a stop list of the stopwords from nltk package
stop_list = set(stopwords.words('english'))

# Loading all the annotations data into a list

annotation = pd.read_csv('H:/consolidate/input_files/annotation_input.csv')
annotation = annotation.drop(annotation.columns[[0]], axis=1)

annotation['Annotation'][annotation['Annotation'].isnull()] = 'None'

annotations = []
a_id = []
for i in range(0, len(annotation)):
    line = annotation.iloc[i, 1]
    a_id.append(str(annotation.iloc[i, 0]).lower())
    lowers = line.lower()
    punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
    annotations.append(punctuation)

# Loading all the bisac data into a list

bisac = pd.read_csv('H:/consolidate/input_files/bisac_input.csv')
bisac = bisac.drop(bisac.columns[[0]], axis=1)


bisac[bisac['Bisac'].isnull()] = 'None'
bisacs = []
for i in range(0, len(bisac)):
    line = bisac.iloc[i, 1]
    bisacl = line.split(';')
    p = ''
    for j in bisacl:
        first = j.split('/')[0].replace(' ', '').replace(',', '~')
        p = p + ' ' + first
        p = p.strip()
    bisacs.append(p)


# Loading all the audience data into a list

audience = pd.read_csv('H:/consolidate/input_files/audience_input.csv')
audience = audience.drop(audience.columns[[0]], axis=1)
audience[audience['Audience'].isnull()] = 'None'

audiences = []

for i in range(0, len(audience)):
    line = audience.iloc[i, 1]
    line = re.sub(' ', '', line)
    lowers = line.lower()
    punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
    audiences.append(punctuation)

# Loading all the series data into a list

series = pd.read_csv('H:/consolidate/input_files/series_input.csv')
series = series.drop(series.columns[[0]], axis=1)


series[series['Series'].isnull()] = ''

seriess = []

for i in range(0, len(series)):
    line = series.iloc[i, 1]
    line = re.sub(' ', '', line)
    lowers = line.lower()
    punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
    seriess.append(punctuation)

# Loading all the author data into a list

author = pd.read_csv('H:/consolidate/input_files/author_input.csv')
author = author.drop(author.columns[[0]], axis=1)


author[author['Author'].isnull()] = ''

authors = []

for i in range(0, len(author)):
    line = author.iloc[i, 1]
    line = re.sub(' ', '', line)
    lowers = line.lower()
    punctuation = lowers.translate(str.maketrans('', '', string.punctuation))
    authors.append(punctuation)

# Loading all the checkouts data into a list

checkout = pd.read_csv('H:/consolidate/input_files/isbn_patron.csv')
# checkout = checkout.drop(checkout.columns[[0]], axis=1)

isbn_counts = checkout.groupby(['ISBN']).size().reset_index(name='counts')

n = 0
j = 0
m = 0
checkouts = []
b_id = []
while j < len(isbn_counts):
    id = isbn_counts.iloc[j, 0]
    b_id.append(str(id).lower())
    count = isbn_counts.iloc[j, 1]
    t = 0
    g = ''
    while t < count:
        line = checkout.iloc[m, 1]
        line = re.sub(',', ' ', line)
        line = re.sub('-', '', line)
        g = g + line + ' '
        m = m + 1
        t = t + 1
    g = g.strip()
    checkouts.append(g)
    if n % 10000 == 0:
        print(n, ' is finished')
    n = n + 1
    j = j + 1

# Loading all the uuids data into a list


# Loading all the uuid data into a list

uuid = pd.read_csv('H:/consolidate/input_files/patron_isbn.csv')
# uuid = uuid.drop(uuid.columns[[0]], axis=1)


uuid_counts = uuid.groupby(['PatronID']).size().reset_index(name='counts')

n = 0
j = 0
m = 0
uuids = []
c_id = []
while j < len(uuid_counts):
    id = uuid_counts.iloc[j, 0]
    c_id.append(str(id).lower())
    count = uuid_counts.iloc[j, 1]
    t = 0
    g = ''
    while t < count:
        line = str(uuid.iloc[m, 1])
        g = g + line + ','
        m = m + 1
        t = t + 1
    g = g.strip(',')
    g = g.split(',')
    uuids.append(g)
    if n % 10000 == 0:
        print(n, ' is finished')
    n = n + 1
    j = j + 1

# Creating bridging tables betweeen the entries from the OMNI table and the CheckoutsTransactionArchive table based on the ISBNs of the books.
bridge_omni = pd.DataFrame({'ISBN': a_id})
bridge_omni['indice_omni'] = bridge_omni.index
bridge_chck = pd.DataFrame({'ISBN': b_id})
bridge_chck['indice_chck'] = bridge_chck.index
chk_omni = pd.merge(bridge_chck, bridge_omni, on='ISBN', how='left')
omni_chk = pd.merge(bridge_omni, bridge_chck, on='ISBN', how='left')
omni_chk['indice_chck'] = pd.to_numeric(omni_chk['indice_chck'], downcast='integer')

# omni_chk.to_csv('H:/consolidate/input_files/omni_bridging_table.csv')


#####-----SEGMENT 2-----#####

from sklearn.feature_extraction.text import CountVectorizer

# creating a count vectorizer for each of our attributes.

# count-vectorizer for annotations
anv = CountVectorizer(max_df=0.85, stop_words=stop_list, max_features=10000)
# count-vectorizer for bisacs
biv = CountVectorizer(max_df=0.85)
# count-vectorizer for audiences
auv = CountVectorizer(max_df=0.85)
# count-vectorizer for series
sev = CountVectorizer(max_df=0.85)
# count-vectorizer for author
autv = CountVectorizer(max_df=0.85)
# count-vectorizer for checkouts
cav = CountVectorizer()

#####-----SEGMENT 3-----#####

# creating the count vector for each attribute

annotations_count_vector = anv.fit_transform(annotations)
bisac_count_vector = biv.fit_transform(bisacs)
audiences_count_vector = auv.fit_transform(audiences)
series_count_vector = sev.fit_transform(seriess)
authors_count_vector = autv.fit_transform(authors)
checkouts_count_vector = cav.fit_transform(checkouts)

# Creating the actual count matrix for each of the attributes

annotationssmatrix = anv.transform(annotations)
bisacsmatrix = biv.transform(bisacs)
audiencesmatrix = auv.transform(audiences)
seriesmatrix = sev.transform(seriess)
authorsmatrix = autv.transform(authors)
checkoutsmatrix = cav.transform(checkouts)

# Checking the size and elements of the vocabulary for each attribute

anvvocab = anv.vocabulary_
bivvocab = biv.vocabulary_
auvvocab = auv.vocabulary_
sevvocab = sev.vocabulary_
autvvocab = autv.vocabulary_
cavvocab = cav.vocabulary_

#####-----SEGMENT 4-----#####

# converting the annotations matrix to the tf-idf format and rounding up the values.

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(annotationssmatrix)

tf_idf_matrix = tfidf.transform(annotationssmatrix)

tf_idf_matrix.data = np.round(tf_idf_matrix.data, 2)




#####-----SEGMENT 5-----#####

# Writing a function which returns the indices of the bridging table for both omni and checkouts

def isbn2index(isbn):
    templist1 = []
    templist2 = []
    for i in isbn:
        if ((omni_chk.loc[omni_chk['ISBN'] == i].empty)):
            continue
        if ((omni_chk.loc[omni_chk['ISBN'] == i, 'indice_chck'].iloc[0]) is None):
            continue
        else:
            templist2.append(int(omni_chk.loc[omni_chk['ISBN'] == i, 'indice_chck'].iloc[0]))
        if ((omni_chk.loc[omni_chk['ISBN'] == i, 'indice_omni'].iloc[0]) is None):
            continue
        else:
            templist1.append(omni_chk.loc[omni_chk['ISBN'] == i, 'indice_omni'].iloc[0])
    return templist1, templist2


# writing a function which takes in the two lists on bridging indices and computes the top n number of books for the user
def booksim(isbn1, isbn2, num_books, uuid):
    # calculating all of the individual similarities for the omni characteristics
    annotationssim = cosine_similarity(tf_idf_matrix[isbn1], tf_idf_matrix)
    bisacsim = cosine_similarity(bisacsmatrix[isbn1], bisacsmatrix)
    audiencesim = cosine_similarity(audiencesmatrix[isbn1], audiencesmatrix)
    seriessim = cosine_similarity(seriesmatrix[isbn1], seriesmatrix)
    authorssim = cosine_similarity(authorsmatrix[isbn1], authorsmatrix)

    # Calculating the similarities for the checkouts
    checkoutssim = cosine_similarity(checkoutsmatrix[isbn2], checkoutsmatrix)
    checkoutssim = np.round(checkoutssim, 2)

    # summing up and averaging all of the omni similarities
    combinedsim = (annotationssim + bisacsim + audiencesim + seriessim + authorssim) / 5
    combinedsim = np.round(combinedsim, 2)

    # creating dataframes to average out all the similarities
    s = pd.DataFrame({'avg_sim': combinedsim.mean(axis=0)})
    s['indice_omni'] = s.index
    t = pd.DataFrame({'checkout_sim': checkoutssim.mean(axis=0)})
    t['indice_chck'] = t.index

    # merging the similarities with the bridging table to combine both checkouts and omni similarities.
    temp2 = pd.merge(omni_chk, s, how='left', on='indice_omni')
    temp2 = pd.merge(temp2, t, how='left', on='indice_chck')
    temp2['checkout_sim'] = temp2['checkout_sim'].fillna(0)

    # Averaging out the similarities without weighing them at this point.
    temp2['total_sim'] = ((temp2['avg_sim'] * 5) + temp2['checkout_sim']) / 6

    # Creating a new varible which has all of the similarities so that we can find n most similar books.
    f = np.array(temp2['total_sim'])
    ind = f.argsort()[-num_books:][::-1]

    # Creating an empty list and storing the top 100 similar books here.
    sim_list = []
    sim_list.append(uuid)
    for d in ind:
        sim_list.append(omni_chk[omni_chk['indice_omni'] == d]['ISBN'].iloc[0])
    return sim_list


#####-----SEGMENT 6-----#####

#fetching user library mapping
inputrows = pd.read_csv('H:/consolidate/input_files/user_library_mapping.csv')
inputrows.columns = ["UUID", "LibraryID"]
user_library_mapping ={}
for i in range(0,len(inputrows)):
    uuid = inputrows.iloc[i,0]
    library_id = inputrows.iloc[i, 1]
    user_library_mapping[uuid] = library_id

#fetching library books checkout count
inputrows = pd.read_csv('H:/consolidate/input_files/library_books_checkout.csv')
inputrows.columns = ["LibraryID", "ISBN", "Checkout count"]
inputrows = inputrows.sort_values(by=["LibraryID", "Checkout count"], ascending=False)
irGrps = inputrows.groupby("LibraryID").head(100)
library_books_mapping = dict()
for index, row in irGrps.iterrows():
    if row["LibraryID"] not in library_books_mapping:
        library_books_mapping[row["LibraryID"]] = list()
        library_books_mapping[row["LibraryID"]].append(row["ISBN"])
collection = get_library_collection()
collection.insert(library_books_mapping)


def get_top_books_users(d):
    user_dict = {}
    for k in range(d[0],d[1]):
        if len(uuids[k])>2:
            cid = c_id[k]
            omnilist, chcklist = isbn2index(uuids[k])
            if (len(omnilist) == 0 or len(chcklist) == 0):
                continue
            toppicks = booksim(omnilist, chcklist, 100, cid)
            user_dict[(c_id[k], user_library_mapping[c_id[k]])] = toppicks
        else:
            user_dict[(c_id[k], user_library_mapping[c_id[k]])] = library_books_mapping[user_library_mapping[c_id[k]]]
    collection = get_user_collection()
    collection.insert(user_dict)


if __name__ == '__main__':
    from multiprocessing import Pool

    batchSize = 10000
    partitions = list()
    for i in range(1, int(len(c_id)/batchSize) + 1):
        partitions.append(((i-1)*batchSize,i*batchSize))
    partitions.append((int(len(c_id) / batchSize) * batchSize, int(len(c_id))))
    p = Pool(10)
    p.map(get_top_books_users, partitions)







