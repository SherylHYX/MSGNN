cd ../src/

../../parallel -j2 --resume-failed --results ../Output/1_A_bitcoin_slashdot_link_sign_direction --joblog ../joblog/1_A_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=1 python ./main_signed_directed_link.py --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/1_B_bitcoin_slashdot_link_sign_direction --joblog ../joblog/1_B_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=1 python ./main_signed_directed_link.py --num_classes 4  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/1_C_bitcoin_slashdot_link_sign_direction --joblog ../joblog/1_C_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=1 python ./main_signed_directed_link.py --num_classes 4 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/1_D_bitcoin_slashdot_link_sign_direction --joblog ../joblog/1_D_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=1 python ./main_signed_directed_link.py --num_classes 4 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/1_E_bitcoin_slashdot_link_sign_direction --joblog ../joblog/1_E_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=1 python ./main_signed_directed_link.py --num_classes 4 --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/1_F_bitcoin_slashdot_link_sign_direction --joblog ../joblog/1_F_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=1 python ./main_signed_directed_link.py --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 0.05 0.1 0.15 0.2
