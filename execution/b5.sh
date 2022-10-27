cd ../src/

../../parallel -j3 --resume-failed --results ../Output/b5SDSBM_gamma --joblog ../joblog/b5_gamma_joblog CUDA_VISIBLE_DEVICES=5 python ./main_SDSBM_node.py --sd_input_feat --weighted_input_feat --N 1000 --size_ratio 1.5 --seed {1}  --runs 2 --p 1 --eta 0 --gamma {2} --dataset 4c --method MSGNN --supervised_loss_ratio 50 --imbalance_loss_ratio 1 --q {3} ::: 10 20 30 40 50 ::: 0 0.05 0.1 0.15 0.2 55 ::: 0. 0.05 0.1 0.15 0.2 0.25

../../parallel -j3 --resume-failed --results ../Output/5_A_finance_link_sign --joblog ../joblog/5_A_finance_link_sign_joblog CUDA_VISIBLE_DEVICES=5 python ./main_link.py --sd_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0 11 22 33 44

../../parallel -j3 --resume-failed --results ../Output/5_B_finance_link_sign --joblog ../joblog/5_B_finance_link_sign_joblog CUDA_VISIBLE_DEVICES=5 python ./main_link.py  --weighted_input_feat --dataset {1} --runs 5 --method MSGNN --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

../../parallel -j3 --resume-failed --results ../Output/5_C_finance_link_sign --joblog ../joblog/5_C_finance_link_sign_joblog CUDA_VISIBLE_DEVICES=5 python ./main_link.py --sd_input_feat  --dataset {1} --runs 5 --method MSGNN --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

../../parallel -j3 --resume-failed --results ../Output/5_D_finance_link_sign --joblog ../joblog/5_D_finance_link_sign_joblog CUDA_VISIBLE_DEVICES=5 python ./main_link.py --weighted_nonnegative_input_feat --weighted_input_feat --dataset {1} --runs 5 --method MSGNN --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0

../../parallel -j3 --resume-failed --results ../Output/5_E_finance_link_sign --joblog ../joblog/5_E_finance_link_sign_joblog CUDA_VISIBLE_DEVICES=5 python ./main_link.py --dataset {1} --runs 5 --method MSGNN --year {2} --q {3} ::: OPCL pvCLCL ::: {2000..2020} ::: 55 0
