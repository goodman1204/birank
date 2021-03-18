import birankpy
import pandas as pd
import sys
import numpy as np
from scipy import stats
import argparse



def read_data(filepath):
    try:
        data = pd.read_csv(filepath)
        # print("loading data ")
    except:
        data = pd.read_csv(filepath,sep='\t')
        # print("loading data ")

    first_column = data.iloc[:, 0]
    second_column = data.iloc[:, 1]

    print("data columns",data.columns)
    print("unique of first column:",data.columns[0],len(first_column.unique()),"unique of second column:",data.columns[1],len(second_column.unique()))

    return data

def calclulate_spearman(a,b):
    corr, _=  stats.spearmanr(a,b)
    return corr, _

def parse_args():
    parser = argparse.ArgumentParser(description="Node clustering")
    parser.add_argument('--ut', type=str, help='user item graph')
    parser.add_argument('--uu', type=str, help='user user graph')
    parser.add_argument('--tt', type=str, help='item item graph')
    parser.add_argument('--gu', type=str, help='user groundtruth ranking')
    parser.add_argument('--gt', type=str, help='item groundtruth ranking')
    parser.add_argument('--model', type=str,default='proposed', help='ranking model')
    # parser.add_argument('--seed', type=int, default=20, help='Random seed.')
    # parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    # parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
    # parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
    # parser.add_argument('--lr', type=float, default=0.002, help='Initial aearning rate.')
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    # parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
    # parser.add_argument('--nClusters',type=int,default=7)
    # parser.add_argument('--num_run',type=int,default=1,help='Number of running times')
    parser.add_argument('--merge_tt', type=int, default=1, help='merge item-item graph.')
    args, unknown = parser.parse_known_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.model not in ['proposed','HITS','CoHITS','BGRM','BiRank']:
        print("model is not defined")
        sys.exit()

    ut= read_data(args.ut)
    uu = read_data(args.uu)
    tt = read_data(args.tt)


    ground_truth_user = read_data(args.gu)

    ground_truth_user.sort_values('num_followers',ascending=False,inplace=True)
    ground_truth_user['num_followers'] = ground_truth_user['num_followers']/sum(ground_truth_user['num_followers'])

    ground_truth_tweet = read_data(args.gt)

    ground_truth_tweet.sort_values('num_favorites_retweets',ascending=False,inplace=True)
    ground_truth_tweet['num_favorites_retweets']=ground_truth_tweet['num_favorites_retweets']/sum(ground_truth_tweet['num_favorites_retweets'])

    columns = ut.columns
    columns_2 = uu.columns
    print("columns names in user-tweet data:",columns)
    print("columns names in user-user data:",columns_2)

    print('tt graph shape',tt.shape)
    bn = birankpy.BipartiteNetwork()

    if args.model =='proposed':
        bn.set_edgelist_two_types(
            ut,
            uu,
            tt,
            top_col=columns[0], bottom_col=columns[1],
            weight_col=None,
            weight_col_2=None,
            weight_col_3=None
        )

        user_birank_df, tweet_birank_df = bn.generate_birank_new(args.merge_tt)



        user_birank_df.sort_values(by=bn.top_col+'_birank', ascending=False,inplace=True)
        tweet_birank_df.sort_values(by=bn.bottom_col+'_birank', ascending=False,inplace=True)
        print(user_birank_df.head(5))
        print(tweet_birank_df.head(5))
    else:
        bn.set_edgelist(
            ut,
            top_col=columns[0], bottom_col=columns[1],
            weight_col=None,
        )
        user_birank_df, tweet_birank_df = bn.generate_birank(normalizer=args.model)
        user_birank_df.sort_values(by=bn.top_col+'_birank', ascending=False,inplace=True)
        tweet_birank_df.sort_values(by=bn.bottom_col+'_birank', ascending=False,inplace=True)

        print(user_birank_df.head(5))
        print(tweet_birank_df.head(5))

    user_number_top_20 = int(ground_truth_user.shape[0]*0.2) # top 20 percent
    # user_number_top_20=100 # top 50
    groundtruth_user_top20 =ground_truth_user.iloc[0:user_number_top_20]['user'].to_list()

    tweet_number_top_20 = int(ground_truth_tweet.shape[0]*0.2) # top 20 percent
    # tweet_number_top_20=100 # top 50
    groundtruth_tweet_top20 =ground_truth_tweet.iloc[0:tweet_number_top_20]['tweet'].to_list()

    predicted_user_top20 = user_birank_df.iloc[0:user_number_top_20]['user'].to_list()
    predicted_tweet_top20 = tweet_birank_df.iloc[0:tweet_number_top_20]['tweet'].to_list()


    common_value_user = set(groundtruth_user_top20).intersection(set(predicted_user_top20))
    top20_accuracy_user = len(common_value_user)/user_number_top_20
    print("top20 accuracy user:",top20_accuracy_user)

    common_value_tweet = set(groundtruth_tweet_top20).intersection(set(predicted_tweet_top20))
    top20_accuracy_tweet = len(common_value_tweet)/tweet_number_top_20
    print("top20 accuracy tweet:",top20_accuracy_tweet)

    #merge ground truth and predicted
    user_merged = pd.merge(ground_truth_user[ground_truth_user['num_followers']>=0],user_birank_df,on='user')
    tweet_merged = pd.merge(ground_truth_tweet[ground_truth_tweet['num_favorites_retweets']>=0],tweet_birank_df,on='tweet')

    print(user_merged.head(5))
    print(tweet_merged.head(5))
    corr,p= calclulate_spearman(user_merged['num_followers'],user_merged['user_birank'])
    print('user spearmanr coefficient',corr,'p-value',p)
    corr,p= calclulate_spearman(tweet_merged['num_favorites_retweets'],tweet_merged['tweet_birank'])
    print('tweet spearmanr coefficient',corr,'p-value',p)


