cd ../src/

../../parallel -j3 --resume-failed --results ../Output/3bfinance_link --joblog ../joblog/3bfinance_link_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --sd_input_feat --weighted_input_feat --year {1}  --runs 5 --dataset OPCL --method MSGNN --num_classes 5 --q 0.25 ::: {2000..2020}

../../parallel -j1 --resume-failed --results ../Output/b3_link_sign --joblog ../joblog/b3_link_sign_joblog CUDA_VISIBLE_DEVICES=3 python ./main_link.py --sd_input_feat --weighted_input_feat --method MSGNN --q 0 --dataset {1} ::: bitcoin_alpha epinions
