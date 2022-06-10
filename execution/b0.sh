cd ../src/

../../parallel -j3 --resume-failed --results ../Output/b0SDSBM_gamma --joblog ../joblog/b0_gamma_joblog CUDA_VISIBLE_DEVICES=0 python ./main_SDSBM_node.py --sd_input_feat --weighted_input_feat --N 1000 --size_ratio 1.5 --seed {1}  --runs 2 --p {2} --eta 0 --gamma {3} --dataset {4} --method MSGNN --supervised_loss_ratio 50 --imbalance_loss_ratio 1 --pbnc_loss_ratio {5} --q 0.25 ::: 10 20 30 40 50 ::: 0.1 1 ::: 0.15 0.2 0.25 ::: 3c 4c ::: 1 0
