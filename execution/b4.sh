cd ../src/

../../parallel -j2 --resume-failed --results ../Output/4_A_bitcoin_epinions_link_direction --joblog ../joblog/4_A_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --direction_only_task --num_classes 5 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0 11 22 33 44

../../parallel -j2 --resume-failed --results ../Output/4_B_bitcoin_epinions_link_direction --joblog ../joblog/4_B_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --direction_only_task --num_classes 5  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/4_C_bitcoin_epinions_link_direction --joblog ../joblog/4_C_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --direction_only_task --num_classes 5 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/4_D_bitcoin_epinions_link_direction --joblog ../joblog/4_D_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --direction_only_task --num_classes 5 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0

../../parallel -j2 --resume-failed --results ../Output/4_E_bitcoin_epinions_link_direction --joblog ../joblog/4_E_bitcoin_epinions_link_direction_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --direction_only_task --num_classes 5 --dataset {1} --runs 5 --method MSGNN  --q {2} ::: epinions bitcoin_alpha ::: 55 0
