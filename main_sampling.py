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

    print("data columns\n",data.columns)
    print("unique of first column:",data.columns[0],len(first_column.unique()),"unique of second column:",data.columns[1],len(second_column.unique()))
    print()

    return data

def topk_computing(ground_truth_user,ground_truth_tweet,user_birank_df,tweet_birank_df,result_values):
    user_number_top_20 = int(ground_truth_user.shape[0]*args.topk) # top k percent
    # user_number_top_20=100 # top 50
    groundtruth_user_top20 =ground_truth_user.iloc[0:user_number_top_20]['user'].to_list()

    tweet_number_top_20 = int(ground_truth_tweet.shape[0]*args.topk) # top k percent
    # tweet_number_top_20=100 # top 50
    groundtruth_tweet_top20 =ground_truth_tweet.iloc[0:tweet_number_top_20]['tweet'].to_list()

    predicted_user_top20 = user_birank_df.iloc[0:user_number_top_20]['user'].to_list()
    predicted_tweet_top20 = tweet_birank_df.iloc[0:tweet_number_top_20]['tweet'].to_list()


    # result_values=[]
    common_value_user = set(groundtruth_user_top20).intersection(set(predicted_user_top20))
    top20_accuracy_user = len(common_value_user)/user_number_top_20
    result_values.append(top20_accuracy_user)
    print("topk:{} accuracy user:".format(args.topk),top20_accuracy_user)

    common_value_tweet = set(groundtruth_tweet_top20).intersection(set(predicted_tweet_top20))
    top20_accuracy_tweet = len(common_value_tweet)/tweet_number_top_20
    result_values.append(top20_accuracy_tweet)
    print("topk:{} accuracy tweet:".format(args.topk),top20_accuracy_tweet)

    return result_values

def calclulate_spearman(a,b,type,result_values):
    corr, p=  stats.spearmanr(a,b)
    print('{} spearmanr coefficient:'.format(type),corr,p)
    result_values.append(corr)
    # result_values.append(p)

    return result_values

def parse_args():
    parser = argparse.ArgumentParser(description="Node clustering")
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--ut', type=str, default='', help='user item graph')
    parser.add_argument('--uu', type=str, default='', help='user user graph')
    parser.add_argument('--tt', type=str, default='', help='item item graph')
    parser.add_argument('--gu', type=str, default='', help='user groundtruth ranking')
    parser.add_argument('--gt', type=str, default='', help='item groundtruth ranking')
    parser.add_argument('--topk', type=float, default=0.01, help='topk pecent of the dataset')
    parser.add_argument('--model', type=str,default='proposed', help='ranking model')
    parser.add_argument('--alpha', type=float, default=0.425, help='alpha')
    parser.add_argument('--delta', type=float, default=0.425, help='delta')
    parser.add_argument('--beta', type=float, default=0.425, help='beta')
    parser.add_argument('--gamma', type=float, default=0.425, help='gamma')
    parser.add_argument('--merge_tt', type=int, default=1, help='merge item-item graph')
    parser.add_argument('--sampling_uu',type=int, default=0,help='if sampling uu graph is required')
    parser.add_argument('--sampling_tt',type=int, default=0,help='if sampling tt graph is required')
    parser.add_argument('--num_run',type=int, default=1,help='repeat experiments')
    parser.add_argument('--verbose',type=bool, default=False,help='detailed output logs')

    args, unknown = parser.parse_known_args()

    return args

def save_results(args,results):

    wp = open('./result_logs/{}_{}_{}'.format(args.model,args.dataset,args.merge_tt),'w')
    wp.write("alpha: {} delta: {} beta: {} gamma: {} topk: {}\n".format(args.alpha,args.delta,args.beta,args.gamma,args.topk))
    wp.write("topk: {1} recommendataion user: {0}\n".format(results[0],args.topk))
    wp.write("topk: {1} recommendataion tweet: {0}\n".format(results[1],args.topk))
    wp.write("spearmanr user corr and p: {}\n".format(results[2]))
    wp.write("spearmanr tweet corr and p: {}\n".format(results[3]))
    wp.write("\n")
    wp.close()


def save_sampling_results(args,overall_results):

    wp = open('./result_logs/{}_{}_{}_sampinguu_{}_samplingtt_{}'.format(args.model,args.dataset,args.merge_tt,args.sampling_uu,args.sampling_tt),'a')

    new_data = np.array(overall_results)

    new_data = np.mean(new_data,axis=1).T

    wp.write("\n\ntopk:{} alpha:{} delta:{} beta:{} gamma:{}\n".format(args.topk,args.alpha,args.delta,args.beta,args.gamma))

    print("data shape:",new_data.shape)
    wp.write("topk user\n")
    for sampling_rate in range(0,21):
        wp.write("{} ".format(new_data[0][sampling_rate]))
    wp.write('\n')

    wp.write("topk tweet\n")
    for sampling_rate in range(0,21):
        wp.write("{} ".format(new_data[1][sampling_rate]))
    wp.write('\n')

    wp.write("spearmanr user\n")
    for sampling_rate in range(0,21):
        wp.write("{} ".format(new_data[2][sampling_rate]))
    wp.write('\n')

    wp.write("spearmanr tweet\n")
    for sampling_rate in range(0,21):
        wp.write("{} ".format(new_data[3][sampling_rate]))
    wp.write('\n')

    wp.close()



if __name__ == '__main__':
    args = parse_args()

    if args.model not in ['proposed','HITS','CoHITS','BGRM','BiRank']:
        print("model is not defined")
        sys.exit()

    args.ut = 'Ranking/{}/{}.ut'.format(args.dataset,args.dataset)
    args.uu = 'Ranking/{}/{}.uu'.format(args.dataset,args.dataset)
    args.tt = 'Ranking/{}/{}.tt'.format(args.dataset,args.dataset)
    args.gu = 'Ranking/{}/{}.gt_user'.format(args.dataset,args.dataset)
    args.gt = 'Ranking/{}/{}.gt_tweet'.format(args.dataset,args.dataset)
    ut= read_data(args.ut)
    uu = read_data(args.uu)
    tt = read_data(args.tt)


    ground_truth_user = read_data(args.gu)

    ground_truth_user.sort_values('num_followers',ascending=False,inplace=True)
    ground_truth_user['num_followers'] = ground_truth_user['num_followers']/sum(ground_truth_user['num_followers'])

    print(ground_truth_user.head())

    ground_truth_tweet = read_data(args.gt)

    ground_truth_tweet.sort_values('num_favorites_retweets',ascending=False,inplace=True)
    ground_truth_tweet['num_favorites_retweets']=ground_truth_tweet['num_favorites_retweets']/sum(ground_truth_tweet['num_favorites_retweets'])
    print(ground_truth_tweet.head())

    columns_ut = ut.columns
    columns_uu = uu.columns
    columns_tt = tt.columns

    print("columns names in user-tweet data:",columns_ut)
    print("columns names in user-user data:",columns_uu)
    print("columns names in item-item data:",columns_tt)

    print('user-item graph shape',ut.shape)
    print('user-user graph shape',uu.shape)
    print('item-item graph shape',tt.shape)


    bn = birankpy.BipartiteNetwork()



    if args.sampling_uu or args.sampling_tt ==1:

        overall_results=[]
        for sampling_rate in range(0,21):

            sampling_rate /=20

            print("sampling rate",sampling_rate)

            uu_temp = uu
            if args.sampling_uu==1 and sampling_rate>0:
                uu_temp = uu.sample(frac=sampling_rate)
                print("uu shape after sampling:",uu_temp.shape)

                args.gamma=0.425
                args.beta = 0.425
                args.alpha = 0.85
                args.delta = 0

            if args.sampling_uu==1 and sampling_rate==0:
                args.gamma=0
                args.beta = 0.85
                args.alpha = 0.85
                args.delta = 0

            tt_temp = tt
            if args.sampling_tt == 1 and sampling_rate>0:
                tt_temp = tt.sample(frac=sampling_rate)
                print("tt shape after sampling:",tt_temp.shape)

                args.gamma=0.0
                args.beta = 0.85
                args.alpha = 0.425
                args.delta = 0.425

            if args.sampling_tt == 1 and sampling_rate==0:
                args.gamma = 0
                args.delta = 0
                args.alpha = 0.85
                args.beta = 0.85

            total_results_per_sampling=[]

            for run in range(args.num_run):

                result_values = []

                if args.model =='proposed':
                    bn.set_edgelist_two_types(
                        ut,
                        uu_temp,
                        tt_temp,
                        top_col=columns_ut[0], bottom_col=columns_ut[1],
                        weight_col=None,
                        weight_col_2=None,
                        weight_col_3=None
                    )

                    user_birank_df, tweet_birank_df = bn.generate_birank_new(args)


                    user_birank_df.sort_values(by=bn.top_col+'_birank', ascending=False,inplace=True)
                    tweet_birank_df.sort_values(by=bn.bottom_col+'_birank', ascending=False,inplace=True)
                    # print(user_birank_df.head(5))
                    # print(tweet_birank_df.head(5))

                topk_computing(ground_truth_user,ground_truth_tweet,user_birank_df,tweet_birank_df,result_values)

                #merge ground truth and predicted
                user_merged = pd.merge(ground_truth_user[ground_truth_user['num_followers']>=0],user_birank_df,on='user')
                tweet_merged = pd.merge(ground_truth_tweet[ground_truth_tweet['num_favorites_retweets']>=0],tweet_birank_df,on='tweet')

                # print(user_merged.head(5))
                # print(tweet_merged.head(5))
                calclulate_spearman(user_merged['num_followers'],user_merged['user_birank'],'user',result_values)

                calclulate_spearman(tweet_merged['num_favorites_retweets'],tweet_merged['tweet_birank'],'tweet',result_values)

                # save_results(args,result_values)
                total_results_per_sampling.append(result_values)

            overall_results.append(total_results_per_sampling)

        save_sampling_results(args,overall_results)



