cd ../src/

../../parallel -j2 --resume-failed --results ../Output/3_A_bitcoin_epinions_link_direction --joblog ../joblog/3_A_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/3_B_bitcoin_epinions_link_direction --joblog ../joblog/3_B_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --num_classes 4  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/3_C_bitcoin_epinions_link_direction --joblog ../joblog/3_C_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/3_D_bitcoin_epinions_link_direction --joblog ../joblog/3_D_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/3_E_bitcoin_epinions_link_direction --joblog ../joblog/3_E_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/3_F_bitcoin_epinions_link_direction --joblog ../joblog/3_F_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=3 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 0.05 0.1 0.15 0.2
