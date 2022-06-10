cd ../src/

../../parallel -j3 --resume-failed --results ../Output/a1SDSBM_eta --joblog ../joblog/a1_eta_joblog CUDA_VISIBLE_DEVICES=1 python ./main_SDSBM_node.py --sd_input_feat --weighted_input_feat --N 1000 --size_ratio 1.5 --seed {1}  --runs 2 --p {2} --gamma 0 --eta {3} --dataset {4} --method MSGNN --supervised_loss_ratio 50 --imbalance_loss_ratio 1 --pbnc_loss_ratio {5} --q 0.25 ::: 10 20 30 40 50 ::: 0.1 1 ::: 0 0.05 0.1 ::: 3c 4c ::: 1 0
