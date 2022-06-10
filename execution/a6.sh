cd ../src/

../../parallel -j3 --resume-failed --results ../Output/a6finance_direction_only_link --joblog ../joblog/a6finance_direction_only_link_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --direction_only_task --sd_input_feat --year {1}  --runs 5 --dataset pvCLCL --method MSGNN --num_classes {2} --q {3} ::: {2000..2020} ::: 4 5 ::: 0.25 0

../../parallel -j3 --resume-failed --results ../Output/a6finance_link --joblog ../joblog/a6finance_link_joblog CUDA_VISIBLE_DEVICES=6 python ./main_signed_directed_link.py --sd_input_feat --year {1}  --runs 5 --dataset pvCLCL --method MSGNN --num_classes {2} --q {3} ::: {2000..2020} ::: 4 5 ::: 0.25 0
