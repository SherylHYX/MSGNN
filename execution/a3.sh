cd ../src/

../../parallel -j3 --resume-failed --results ../Output/3afinance_link --joblog ../joblog/3afinance_link_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --sd_input_feat --weighted_input_feat --year {1}  --runs 5 --dataset OPCL --method MSGNN --num_classes 4 --q 0.25 ::: {2000..2020}

../../parallel -j2 --resume-failed --results ../Output/3a_bitcoin_direction_only_link --joblog ../joblog/3a_bitcoin_direction_only_link_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --sd_input_feat --weighted_input_feat --dataset bitcoin_otc --runs 5 --method MSGNN --num_classes {1} --q 0.25 ::: 4 5

../../parallel -j2 --resume-failed --results ../Output/3a_bitcoin_link --joblog ../joblog/3a_bitcoin_link_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --sd_input_feat --weighted_input_feat --dataset bitcoin_otc --runs 5 --method MSGNN --num_classes {1} --q 0.25 ::: 4 5