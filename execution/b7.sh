cd ../src/

../../parallel -j1 --resume-failed --results ../Output/b7_SSBM10000 --joblog ../joblog/b7_SSBM10000_joblog CUDA_VISIBLE_DEVICES=7 python ./main_SSSNET_node.py --dataset SSBM --N 10000 --size_ratio 1.5 --p 0.01 --K 5 --runs 2 --seed {1} --method MSGNN --eta {2} ::: 10 20 30 40 50 ::: 0 0.05 0.1 0.15 0.2 55 0.3 0.35 0.4
