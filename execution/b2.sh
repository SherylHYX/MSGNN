cd ../src/

../../parallel -j3 --resume-failed --results ../Output/2bfinance_link --joblog ../joblog/2bfinance_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --sd_input_feat --weighted_input_feat --year {1}  --runs 5 --dataset OPCL --method MSGNN --num_classes 5 --q 0.25 ::: {2000..2020}

../../parallel -j1 --resume-failed --results ../Output/b2_link_sign --joblog ../joblog/b2_link_sign_joblog CUDA_VISIBLE_DEVICES=2 python ./main_link.py --sd_input_feat --weighted_input_feat --method MSGNN --q 0 --dataset {1} ::: bitcoin_otc slashdot
