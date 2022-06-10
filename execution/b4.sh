cd ../src/

../../parallel -j3 --resume-failed --results ../Output/b4finance_direction_only_link --joblog ../joblog/b4finance_direction_only_link_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --direction_only_task --year {1}  --runs 5 --dataset pvCLCL --method MSGNN --num_classes {2} --q {3} ::: {2000..2020} ::: 4 5 ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/b4finance_link --joblog ../joblog/b4finance_link_joblog CUDA_VISIBLE_DEVICES=4 python ./main_signed_directed_link.py --year {1}  --runs 5 --dataset pvCLCL --method MSGNN --num_classes {2} --q {3} ::: {2000..2020} ::: 4 5 ::: 0.25 0
