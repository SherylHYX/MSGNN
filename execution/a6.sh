cd ../src/

../../parallel -j3 --resume-failed --results ../Output/a6SDSBM_gamma --joblog ../joblog/a6_gamma_joblog CUDA_VISIBLE_DEVICES=6 python ./main_SDSBM_node.py --sd_input_feat --weighted_input_feat --N 1000 --size_ratio 1.5 --seed {1}  --runs 2 --p 0.1 --eta 0 --gamma {2} --dataset {3} --method MSGNN --supervised_loss_ratio 50 --imbalance_loss_ratio 1 --q {4} ::: 10 20 30 40 50 ::: 0 0.05 0.1 0.15 0.2 55 ::: 3c 4c ::: 0. 0.05 0.1 0.15 0.2 0.25

../../parallel -j3 --resume-failed --results ../Output/6_A_finance_link_direction --joblog ../joblog/6_A_finance_link_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0 11 22 33 44

../../parallel -j3 --resume-failed --results ../Output/6_B_finance_link_direction --joblog ../joblog/6_B_finance_link_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --direction_only_task --num_classes 4  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

../../parallel -j3 --resume-failed --results ../Output/6_C_finance_link_direction --joblog ../joblog/6_C_finance_link_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

../../parallel -j3 --resume-failed --results ../Output/6_D_finance_link_direction --joblog ../joblog/6_D_finance_link_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

../../parallel -j3 --resume-failed --results ../Output/6_E_finance_link_direction --joblog ../joblog/6_E_finance_link_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --direction_only_task --num_classes 4 --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

