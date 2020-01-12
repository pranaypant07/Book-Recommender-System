'''
Implementing top books for different categories in this code.
Reducing run time by using just one for loop for all the preprocessing
'''



import pandas as pd
import re
import string
import numpy as np
from nltk.corpus import stopwords

stop_list = set(stopwords.words('english'))

def get_sim_matrices():
    features_df = pd.read_csv('book_char_sample_20000.csv')
    features_df = features_df.drop(features_df.columns[[0]], axis=1)
    features_df = features_df.head(5000)


    features_df = features_df.replace(np.nan, 'None', regex=True)

    annotation_children = []
    audiences_children = []
    author_children = []
    series_children = []
    bisac_children = []


    annotation_scholarly = []
    audiences_scholarly = []
    author_scholarly = []
    series_scholarly = []
    bisac_scholarly = []


    annotation_teen = []
    audiences_teen = []
    author_teen = []
    series_teen = []
    bisac_teen = []

    annotation_general = []
    audiences_general = []
    author_general = []
    series_general = []
    bisac_general = []


    isbn_children = []
    isbn_scholarly = []
    isbn_teen = []
    isbn_general = []

    for i in range(0, len(features_df)):

        series = features_df.iloc[i, 2]
        author = features_df.iloc[i, 3]
        bisac = features_df.iloc[i, 6]
        audience = features_df.iloc[i, 7]
        annotation = features_df.iloc[i, 8]

        if ';' in bisac:
            bisacl = bisac.split(';')
            bisacl = bisacl[0].split('/')
        else:
            bisacl = bisac.split('/')
        bisacl = bisacl[0]


        series = re.sub(' ', '', series)
        series = series.lower()
        author = re.sub(' ', '', author)
        author = author.lower()
        audience = re.sub(' ', '', audience)
        audience = audience.lower()
        annotation = re.sub(' ', '', annotation)
        annotation = annotation.lower()

        series_punctuation = series.translate(str.maketrans('', '', string.punctuation))
        author_punctuation = author.translate(str.maketrans('', '', string.punctuation))
        audience_punctuation = audience.translate(str.maketrans('', '', string.punctuation))
        annotation_punctuation = annotation.translate(str.maketrans('', '', string.punctuation))
        bisac_punctuation = bisacl.translate(str.maketrans('', '', string.punctuation))


        if 'general' or 'none' in audience_punctuation:
            isbn_general.append(features_df.iloc[i, 0])
            annotation_general.append(annotation_punctuation)
            audiences_general.append(audience_punctuation)
            series_general.append(series_punctuation)
            author_general.append(author_punctuation)
            bisac_general.append(bisac_punctuation)
        if 'teen' in audience_punctuation:
            isbn_teen.append(features_df.iloc[i, 0])
            annotation_teen.append(annotation_punctuation)
            audiences_teen.append(audience_punctuation)
            series_teen.append(series_punctuation)
            author_teen.append(author_punctuation)
            bisac_teen.append(bisac_punctuation)
        if 'scholarly' in audience_punctuation:
            isbn_scholarly.append(features_df.iloc[i, 0])
            annotation_scholarly.append(annotation_punctuation)
            audiences_scholarly.append(audience_punctuation)
            series_scholarly.append(series_punctuation)
            author_scholarly.append(author_punctuation)
            bisac_scholarly.append(bisac_punctuation)
        if 'children' in audience_punctuation:
            isbn_children.append(features_df.iloc[i, 0])
            annotation_children.append(annotation_punctuation)
            audiences_children.append(audience_punctuation)
            series_children.append(series_punctuation)
            author_children.append(author_punctuation)
            bisac_children.append(bisac_punctuation)

    #####-----SEGMENT 2-----#####

    from sklearn.feature_extraction.text import CountVectorizer

    # creating a count vectorizer for each of our attributes.


    anv_general = CountVectorizer(max_df=0.85, stop_words=stop_list, max_features=10000)
    anv_teen = CountVectorizer(max_df=0.85, stop_words=stop_list, max_features=10000)
    anv_scholarly = CountVectorizer(max_df=0.85, stop_words=stop_list, max_features=10000)
    anv_children = CountVectorizer(max_df=0.85, stop_words=stop_list, max_features=10000)

    biv_general = CountVectorizer(max_df=0.85)
    biv_teen = CountVectorizer(max_df=0.85)
    biv_scholarly = CountVectorizer(max_df=0.85)
    biv_children = CountVectorizer(max_df=0.85)


    auv_general = CountVectorizer(max_df=0.85)
    auv_teen = CountVectorizer(max_df=0.85)
    auv_scholarly = CountVectorizer()
    auv_children = CountVectorizer(max_df=0.85)


    sev_general = CountVectorizer(max_df=0.85)
    sev_teen = CountVectorizer(max_df=0.85)
    sev_scholarly = CountVectorizer(max_df=0.85)
    sev_children = CountVectorizer(max_df=0.85)


    autv_general = CountVectorizer(max_df=0.85)
    autv_teen = CountVectorizer(max_df=0.85)
    autv_scholarly = CountVectorizer(max_df=0.85)
    autv_children = CountVectorizer(max_df=0.85)


    #####-----SEGMENT 3-----#####

    # creating the count vector for each attribute

    annotations_general_count_vector = anv_general.fit_transform(annotation_general)
    annotations_teen_count_vector = anv_teen.fit_transform(annotation_teen)
    annotations_scholarly_count_vector = anv_scholarly.fit_transform(annotation_scholarly)
    annotations_children_count_vector = anv_children.fit_transform(annotation_children)



    bisac_general_count_vector = biv_general.fit_transform(bisac_general)
    bisac_teen_count_vector = biv_teen.fit_transform(bisac_teen)
    bisac_scholarly_count_vector = biv_scholarly.fit_transform(bisac_scholarly)
    bisac_children_count_vector = biv_children.fit_transform(bisac_children)



    audiences_general_count_vector = auv_general.fit_transform(audiences_general)
    audiences_teen_count_vector = auv_teen.fit_transform(audiences_teen)
    audiences_scholarly_count_vector = auv_scholarly.fit_transform(audiences_scholarly)
    audiences_children_count_vector = auv_children.fit_transform(audiences_children)

    series_general_count_vector = sev_general.fit_transform(series_general)
    series_teen_count_vector = sev_teen.fit_transform(series_teen)
    series_scholarly_count_vector = sev_scholarly.fit_transform(series_scholarly)
    series_children_count_vector = sev_children.fit_transform(series_children)


    authors_general_count_vector = autv_general.fit_transform(author_general)
    authors_teen_count_vector = autv_teen.fit_transform(author_teen)
    authors_scholarly_count_vector = autv_scholarly.fit_transform(author_scholarly)
    authors_children_count_vector = autv_children.fit_transform(author_children)



    # Creating the actual count matrix for each of the attributes

    annotationssmatrix_general = anv_general.transform(annotation_general)
    annotationssmatrix_teen = anv_teen.transform(annotation_teen)
    annotationssmatrix_scholarly = anv_scholarly.transform(annotation_scholarly)
    annotationssmatrix_children = anv_children.transform(annotation_children)

    bisacsmatrix_general = biv_general.transform(bisac_general)
    bisacsmatrix_teen = biv_teen.transform(bisac_teen)
    bisacsmatrix_scholarly = biv_scholarly.transform(bisac_scholarly)
    bisacsmatrix_children = biv_children.transform(bisac_children)



    audiencesmatrix_general = auv_general.transform(audiences_general)
    audiencesmatrix_teen = auv_teen.transform(audiences_teen)
    audiencesmatrix_scholarly = auv_scholarly.transform(audiences_scholarly)
    audiencesmatrix_children = auv_children.transform(audiences_children)



    seriesmatrix_general = sev_general.transform(series_general)
    seriesmatrix_teen = sev_teen.transform(series_teen)
    seriesmatrix_scholarly = sev_scholarly.transform(series_scholarly)
    seriesmatrix_children = sev_children.transform(series_children)



    authorsmatrix_general = autv_general.transform(author_general)
    authorsmatrix_teen = autv_teen.transform(author_teen)
    authorsmatrix_scholarly = autv_scholarly.transform(author_scholarly)
    authorsmatrix_children = autv_children.transform(author_children)


    # # Checking the size and elements of the vocabulary for each attribute
    #
    # anvvocab = anv.vocabulary_
    # bivvocab = biv.vocabulary_
    # auvvocab = auv.vocabulary_
    # sevvocab = sev.vocabulary_
    # autvvocab = autv.vocabulary_


    #####-----SEGMENT 4-----#####

    # converting the annotations matrix to the tf-idf format and rounding up the values.

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_general = TfidfTransformer(norm="l2")
    tfidf_teen = TfidfTransformer(norm="l2")
    tfidf_scholarly = TfidfTransformer(norm="l2")
    tfidf_children = TfidfTransformer(norm="l2")


    tfidf_general.fit(annotationssmatrix_general)
    tfidf_teen.fit(annotationssmatrix_teen)
    tfidf_scholarly.fit(annotationssmatrix_scholarly)
    tfidf_children.fit(annotationssmatrix_children)

    tf_idf_matrix_general = tfidf_general.transform(annotationssmatrix_general)
    tf_idf_matrix_teen = tfidf_teen.transform(annotationssmatrix_teen)
    tf_idf_matrix_scholarly = tfidf_scholarly.transform(annotationssmatrix_scholarly)
    tf_idf_matrix_children = tfidf_children.transform(annotationssmatrix_children)

    tf_idf_matrix_general.data = np.round(tf_idf_matrix_general.data, 2)
    tf_idf_matrix_teen.data = np.round(tf_idf_matrix_teen.data, 2)
    tf_idf_matrix_scholarly.data = np.round(tf_idf_matrix_scholarly.data, 2)
    tf_idf_matrix_children.data = np.round(tf_idf_matrix_children.data, 2)

    annotationssim_general = cosine_similarity(tf_idf_matrix_general, tf_idf_matrix_general)
    annotationssim_teen = cosine_similarity(tf_idf_matrix_teen, tf_idf_matrix_teen)
    annotationssim_scholarly = cosine_similarity(tf_idf_matrix_scholarly, tf_idf_matrix_scholarly)
    annotationssim_children = cosine_similarity(tf_idf_matrix_children, tf_idf_matrix_children)


    bisacsim_general = cosine_similarity(bisacsmatrix_general, bisacsmatrix_general)
    bisacsim_teen = cosine_similarity(bisacsmatrix_teen, bisacsmatrix_teen)
    bisacsim_scholarly = cosine_similarity(bisacsmatrix_scholarly, bisacsmatrix_scholarly)
    bisacsim_children = cosine_similarity(bisacsmatrix_children, bisacsmatrix_children)


    audiencesim_general = cosine_similarity(audiencesmatrix_general, audiencesmatrix_general)
    audiencesim_teen = cosine_similarity(audiencesmatrix_teen, audiencesmatrix_teen)
    audiencesim_scholarly = cosine_similarity(audiencesmatrix_scholarly, audiencesmatrix_scholarly)
    audiencesim_children = cosine_similarity(audiencesmatrix_children, audiencesmatrix_children)

    seriessim_general = cosine_similarity(seriesmatrix_general, seriesmatrix_general)
    seriessim_teen = cosine_similarity(seriesmatrix_teen, seriesmatrix_teen)
    seriessim_scholarly = cosine_similarity(seriesmatrix_scholarly, seriesmatrix_scholarly)
    seriessim_children = cosine_similarity(seriesmatrix_children, seriesmatrix_children)


    authorssim_general = cosine_similarity(authorsmatrix_general, authorsmatrix_general)
    authorssim_teen = cosine_similarity(authorsmatrix_teen, authorsmatrix_teen)
    authorssim_scholarly = cosine_similarity(authorsmatrix_scholarly, authorsmatrix_scholarly)
    authorssim_children = cosine_similarity(authorsmatrix_children, authorsmatrix_children)

    return