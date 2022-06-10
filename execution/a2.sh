cd ../src/

../../parallel -j3 --resume-failed --results ../Output/2afinance_link --joblog ../joblog/2afinance_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --sd_input_feat --weighted_input_feat --year {1}  --runs 5 --dataset OPCL --method MSGNN --num_classes 4 --q 0.25 ::: {2000..2020}

../../parallel -j2 --resume-failed --results ../Output/2a_A_bitcoin_direction_only_link --joblog ../joblog/2a_A_bitcoin_direction_only_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --direction_only_task --sd_input_feat --weighted_input_feat --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_A_bitcoin_link --joblog ../joblog/2a_A_bitcoin_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --sd_input_feat --weighted_input_feat --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_B_bitcoin_direction_only_link --joblog ../joblog/2a_B_bitcoin_direction_only_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --direction_only_task  --weighted_input_feat --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_B_bitcoin_link --joblog ../joblog/2a_B_bitcoin_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py  --weighted_input_feat --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_C_bitcoin_direction_only_link --joblog ../joblog/2a_C_bitcoin_direction_only_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --direction_only_task --sd_input_feat  --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_C_bitcoin_link --joblog ../joblog/2a_C_bitcoin_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --sd_input_feat  --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_D_bitcoin_direction_only_link --joblog ../joblog/2a_D_bitcoin_direction_only_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --direction_only_task --weighted_nonnegative_input_feat --weighted_input_feat --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_D_bitcoin_link --joblog ../joblog/2a_D_bitcoin_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --weighted_nonnegative_input_feat --weighted_input_feat --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_E_bitcoin_direction_only_link --joblog ../joblog/2a_E_bitcoin_direction_only_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --direction_only_task --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

../../parallel -j2 --resume-failed --results ../Output/2a_E_bitcoin_link --joblog ../joblog/2a_E_bitcoin_link_joblog CUDA_VISIBLE_DEVICES=2 python ./main_signed_directed_link.py --dataset bitcoin_alpha --runs 5 --method MSGNN --num_classes {1} --q {2} ::: 4 5 ::: 0.25 0

