import birankpy
import pandas as pd
import sys


def read_data(filepath):
    try:
        data = pd.read_csv(filepath)
        print("loading data ")
    except:
        data = pd.read_csv(filepath,sep='\t')
        print("loading data ")

    first_column = data.iloc[:, 0]
    second_column = data.iloc[:, 1]

    print("unique of first column:",len(first_column.unique()))
    print("unique of second column:",len(second_column.unique()))

    return data

if __name__=='__main__':

    filepath = sys.argv[1] # user to tweet edges
    filepath_user_user = sys.argv[2] # user to user edges

    data = read_data(filepath)
    data_2 = read_data(filepath_user_user)

    columns = data.columns
    columns_2 = data_2.columns
    print("columns names in user-tweet data:",columns)
    print("columns names in user-user data:",columns_2)

    bn = birankpy.BipartiteNetwork()
    bn.set_edgelist_two_types(
        data,
        data_2,
        top_col=columns[0], bottom_col=columns[1],
        weight_col=None,
        weight_col_2=None
    )

    user_birank_df, tweet_birank_df = bn.generate_birank_new()
    print(user_birank_df.sort_values(by=bn.top_col+'_birank', ascending=False))
    print(tweet_birank_df.sort_values(by=bn.bottom_col+'_birank', ascending=False))
    # char_birank_df, _ = bn.generate_birank(normalizer='CoHITS')
    # char_birank_df.sort_values(by='character_birank', ascending=False).head()
    # un = bn.unipartite_projection(on='character')
    # char_projected_pagerank_df = un.generate_pagerank()
    # char_projected_pagerank_df.sort_values(by='pagerank', ascending=False).head()


