cd ../src/

../../parallel -j3 --resume-failed --results ../Output/a5SDSBM_gamma --joblog ../joblog/a5_gamma_joblog CUDA_VISIBLE_DEVICES=5 python ./main_SDSBM_node.py --sd_input_feat --weighted_input_feat --N 1000 --size_ratio 1.5 --seed {1}  --runs 2 --p 1 --eta 0 --gamma {2} --dataset 3c --method MSGNN --supervised_loss_ratio 50 --imbalance_loss_ratio 1 --q {3} ::: 10 20 30 40 50 ::: 0 0.05 0.1 0.15 0.2 0.25 ::: 0. 0.05 0.1 0.15 0.2 0.25

../../parallel -j3 --resume-failed --results ../Output/5_A_finance_link_sign_direction --joblog ../joblog/5_A_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=5 python ./main_signed_directed_link.py --num_classes 5 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/5_B_finance_link_sign_direction --joblog ../joblog/5_B_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=5 python ./main_signed_directed_link.py --num_classes 5  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/5_C_finance_link_sign_direction --joblog ../joblog/5_C_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=5 python ./main_signed_directed_link.py --num_classes 5 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/5_D_finance_link_sign_direction --joblog ../joblog/5_D_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=5 python ./main_signed_directed_link.py --num_classes 5 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/5_E_finance_link_sign_direction --joblog ../joblog/5_E_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=5 python ./main_signed_directed_link.py --num_classes 5 --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/5_F_finance_link_sign_direction --joblog ../joblog/5_F_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=5 python ./main_signed_directed_link.py --num_classes 5 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.05 0.1 0.15 0.2
