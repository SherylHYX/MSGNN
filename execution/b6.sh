cd ../src/

../../parallel -j1 --resume-failed --results ../Output/b6_POLSSBM5000 --joblog ../joblog/b6_POLSSBM5000_joblog CUDA_VISIBLE_DEVICES=6 python ./main_SSSNET_node.py --dataset polarized --N 500 --total_n 5000 --size_ratio 1.5 --p 0.1 --K 2 --num_com 5  --runs 2 --seed {1} --method MSGNN --eta {2} ::: 10 20 30 40 50  ::: 0 0.05 0.1 0.15 0.2 0.25

../../parallel -j3 --resume-failed --results ../Output/6_A_finance_link_sign_direction --joblog ../joblog/6_A_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/6_B_finance_link_sign_direction --joblog ../joblog/6_B_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --num_classes 4  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/6_C_finance_link_sign_direction --joblog ../joblog/6_C_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --num_classes 4 --sd_input_feat  --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/6_D_finance_link_sign_direction --joblog ../joblog/6_D_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --num_classes 4 --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/6_E_finance_link_sign_direction --joblog ../joblog/6_E_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --num_classes 4 --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/6_F_finance_link_sign_direction --joblog ../joblog/6_F_finance_link_sign_direction_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --num_classes 4 --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN  --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 0.05 0.1 0.15 0.2
