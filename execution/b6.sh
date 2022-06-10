cd ../src/

../../parallel -j1 --resume-failed --results ../Output/b6_POLSSBM5000 --joblog ../joblog/b6_POLSSBM5000_joblog CUDA_VISIBLE_DEVICES=6 python ./main_SSSNET_node.py --dataset polarized --N 500 --total_n 5000 --size_ratio 1.5 --p 0.1 --K 2 --num_com 5  --runs 2 --seed {1} --method MSGNN --eta {2} ::: 10 20 30 40 50  ::: 0 0.05 0.1 0.15 0.2 0.25
