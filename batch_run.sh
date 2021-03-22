#! /bin/bash
declare -a dataset=("gossipcop_fake" "gossipcop_real" "politifact_fake" "politifact_real" "Aminer")

declare -a methods=("proposed" "HITS" "CoHITS" "BGRM" "BiRank")

#IFS=,

#"${arr[@]}"

for d in ${dataset[@]};  
do
    echo $d
	for m in ${methods[@]};
	do 
        python main.py   --model $m --merge_tt 0 --dataset $d
        echo $m
	done 
done

for d in ${dataset[@]};  
do
        echo $d
        python main.py   --model proposed --merge_tt 1 --dataset $d
done
