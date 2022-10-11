cd ../src/

../../parallel -j2 --resume-failed --results ../Output/0_A_bitcoin_slashdot_link_sign --joblog ../joblog/0_A_bitcoin_slashdot_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/0_B_bitcoin_slashdot_link_sign --joblog ../joblog/0_B_bitcoin_slashdot_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/0_C_bitcoin_slashdot_link_sign --joblog ../joblog/0_C_bitcoin_slashdot_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/0_D_bitcoin_slashdot_link_sign --joblog ../joblog/0_D_bitcoin_slashdot_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/0_E_bitcoin_slashdot_link_sign --joblog ../joblog/0_E_bitcoin_slashdot_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/0_F_bitcoin_slashdot_link_sign --joblog ../joblog/0_F_bitcoin_slashdot_link_sign_joblog CUDA_VISIBLE_DEVICES=0 python ./main_link.py --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.05 0.1 0.15 0.2
