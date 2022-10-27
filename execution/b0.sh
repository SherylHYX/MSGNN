cd ../src/

../../parallel -j2 --resume-failed --results ../Output/0_A_bitcoin_epinions_link_sign --joblog ../joblog/0_A_bitcoin_epinions_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0 11 22 33 44

../../parallel -j2 --resume-failed --results ../Output/0_B_bitcoin_epinions_link_sign --joblog ../joblog/0_B_bitcoin_epinions_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/0_C_bitcoin_epinions_link_sign --joblog ../joblog/0_C_bitcoin_epinions_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/0_D_bitcoin_epinions_link_sign --joblog ../joblog/0_D_bitcoin_epinions_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/0_E_bitcoin_epinions_link_sign --joblog ../joblog/0_E_bitcoin_epinions_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0
