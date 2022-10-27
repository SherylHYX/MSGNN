cd ../src/

../../parallel -j2 --resume-failed --results ../Output/2_A_bitcoin_slashdot_link_sign_direction --joblog ../joblog/2_A_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --num_classes 5 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 55 0 11 22 33 44

../../parallel -j2 --resume-failed --results ../Output/2_B_bitcoin_slashdot_link_sign_direction --joblog ../joblog/2_B_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --num_classes 5  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/2_C_bitcoin_slashdot_link_sign_direction --joblog ../joblog/2_C_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --num_classes 5 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/2_D_bitcoin_slashdot_link_sign_direction --joblog ../joblog/2_D_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --num_classes 5 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/2_E_bitcoin_slashdot_link_sign_direction --joblog ../joblog/2_E_bitcoin_slashdot_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --num_classes 5 --dataset {1} --runs 5 --method MSGNN  --q {2} ::: slashdot bitcoin_otc ::: 55 0
