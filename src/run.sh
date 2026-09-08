# python3 main_ours.py -nsp -mask
# python3 main_ours.py -t -l=log0
# python3 main_ours.py -nsp -mask --weight_decay 0.05
# python3 main_ours.py -t -l=log2
# python3 main_ours.py -nsp -mask --weight_decay 0.001
# python3 main_ours.py -t -l=log4
# python3 main_ours.py -nsp -mask --weight_decay 0.005
# python3 main_ours.py -t -l=log6
# python3 main_ours.py -nsp -mask --weight_decay 0.02
# python3 main_ours.py -t -l=log8
# python3 main_ours.py -nsp -mask --weight_decay 0.03
# python3 main_ours.py -t -l=log10
# python3 main_ours.py -nsp -mask --weight_decay 0.04
# python3 main_ours.py -t -l=log12
# python3 main_ours.py -nsp -mask --weight_decay 0.005
# python3 main_ours.py -t -l=log6

# python3 main_ours.py -nsp -mask --weight_decay 0.05 --lr 1.1e-5
# python3 main_ours.py -t -l=log0
python3 main_ours.py -nsp -mask --weight_decay 0.055  --lr 2e-5 --model_name RCAMed1 --weight_ddi 0.75
python3 main_ours.py -t -l=log4 --model_name RCAMed1
python3 main_ours.py -nsp -mask --weight_decay 0.045  --lr 2e-5 --model_name RCAMed1 --weight_ddi 0.75
python3 main_ours.py -t -l=log6 --model_name RCAMed1
python3 main_ours.py -nsp -mask --weight_decay 0.05 --dropout 0.25 --lr 2e-5 --model_name RCAMed1 --weight_ddi 0.75
python3 main_ours.py -t -l=log8 --model_name RCAMed1
python3 main_ours.py -nsp -mask --weight_decay 0.045  --dropout 0.35 --lr 2e-5 --model_name RCAMed1 --weight_ddi 0.75
python3 main_ours.py -t -l=log10 --model_name RCAMed1