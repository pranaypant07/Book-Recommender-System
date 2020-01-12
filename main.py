import pandas as pd
import Recommenders
from sklearn.model_selection import train_test_split


# triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt'
# songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv'



df = pd.read_csv('checkout_input.csv')

test = pd.crosstab(df.UUID, df.ISBN)
print test

# mux = pandas.MultiIndex.from_product(s.index.levels, names=s.index.names)
# df = s.reindex(mux, fill_value=0).reset_index(name='count')

# print df
# print len(df['UUID'].unique())
# print len(df['ISBN'].unique())

# train_data, test_data = train_test_split(df, test_size = 0.20, random_state=0)
# print '************'
# users = df['UUID'].unique()
# is_model = Recommenders.item_similarity_recommender_py()
# is_model.create(df,'UUID','ISBN')
#
# user_id = 'd4d3108e-0871-4a32-9035-5a040053f279'
# print '***************'
# user_items = is_model.get_user_items(user_id)
#
# is_model.recommend(user_id)