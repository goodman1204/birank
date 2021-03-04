import birankpy
import pandas as pd
import sys
import numpy as np


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

    if len(sys.argv)<=5:
        print("parameter errors\n please input data1 data2 method[proposed,HITS,CoHITS,BGRM,BiRank] groundtruth_user groundtruth_tweet ")
        sys.exit()

    filepath = sys.argv[1] # user to tweet edges
    filepath_user_user = sys.argv[2] # user to user edges
    method = sys.argv[3]
    if method not in ['proposed','HITS','CoHITS','BGRM','BiRank']:
        print("method is not defined")
        sys.exit()

    data = read_data(filepath)
    data_2 = read_data(filepath_user_user)
    ground_truth_user = read_data(sys.argv[4])
    ground_truth_user.sort_values('num_followers',ascending=False,inplace=True)
    ground_truth_tweet = read_data(sys.argv[5])
    ground_truth_tweet.sort_values('num_favorites_retweets',ascending=False,inplace=True)

    columns = data.columns
    columns_2 = data_2.columns
    print("columns names in user-tweet data:",columns)
    print("columns names in user-user data:",columns_2)

    bn = birankpy.BipartiteNetwork()

    if method =='proposed':
        bn.set_edgelist_two_types(
            data,
            data_2,
            top_col=columns[0], bottom_col=columns[1],
            weight_col=None,
            weight_col_2=None
        )

        user_birank_df, tweet_birank_df = bn.generate_birank_new()
        user_birank_df.sort_values(by=bn.top_col+'_birank', ascending=False,inplace=True)
        tweet_birank_df.sort_values(by=bn.bottom_col+'_birank', ascending=False,inplace=True)
        print(user_birank_df)
        print(tweet_birank_df)
    else:
        bn.set_edgelist(
            data,
            top_col=columns[0], bottom_col=columns[1],
            weight_col=None,
        )
        user_birank_df, tweet_birank_df = bn.generate_birank(normalizer=method)
        user_birank_df.sort_values(by=bn.top_col+'_birank', ascending=False,inplace=True)
        tweet_birank_df.sort_values(by=bn.bottom_col+'_birank', ascending=False,inplace=True)

        print(user_birank_df)
        print(tweet_birank_df)

    user_number_top_20 = int(ground_truth_user.shape[0]*0.2)
    groundtruth_user_top20 =ground_truth_user.iloc[0:user_number_top_20]['user'].to_list()

    tweet_number_top_20 = int(ground_truth_tweet.shape[0]*0.2)
    groundtruth_tweet_top20 =ground_truth_tweet.iloc[0:tweet_number_top_20]['tweet'].to_list()

    predicted_user_top20 = user_birank_df.iloc[0:user_number_top_20]['user'].to_list()
    predicted_tweet_top20 = tweet_birank_df.iloc[0:tweet_number_top_20]['tweet'].to_list()


    common_value_user = set(groundtruth_user_top20).intersection(set(predicted_user_top20))
    top20_accuracy_user = len(common_value_user)/user_number_top_20
    print("top20 accuracy user:",top20_accuracy_user)

    common_value_tweet = set(groundtruth_tweet_top20).intersection(set(predicted_tweet_top20))
    top20_accuracy_tweet = len(common_value_tweet)/tweet_number_top_20
    print("top20 accuracy tweet:",top20_accuracy_tweet)
    # char_birank_df, _ = bn.generate_birank(normalizer='CoHITS')
    # char_birank_df.sort_values(by='character_birank', ascending=False).head()
    # un = bn.unipartite_projection(on='character')
    # char_projected_pagerank_df = un.generate_pagerank()
    # char_projected_pagerank_df.sort_values(by='pagerank', ascending=False).head()


