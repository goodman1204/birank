#! /bin/bash
declare -a dataset=("gossipcop_fake" "gossipcop_real" "politifact_fake" "politifact_real" "Aminer")

declare -a methods=("proposed" "HITS" "CoHITS" "BGRM" "BiRank")



for d in ${dataset[@]};  
do
    echo $d
    python main_parameter_analysis.py --para 1  --model proposed --merge_tt 0 --dataset $d --testing_alpha 1 --verbose  1
    python main_parameter_analysis.py --para 1  --model proposed --merge_tt 1 --dataset $d --testing_alpha 1 --verbose  1
done
