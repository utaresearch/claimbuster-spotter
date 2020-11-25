# Copyright (C) 2020 IDIR Lab - UT Arlington
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License v3 as published by
#     the Free Software Foundation.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact Information:
#     See: https://idir.uta.edu/cli.html
#
#     Chengkai Li
#     Box 19015
#     Arlington, TX 76019
#

nohup sh -c "python3 train.py --cs_adv_train=False --cs_model_size=base --cs_model_dir=output --cs_gpu=0 --cs_weight_classes_loss=True"&>nohup.out&

nohup sh -c ""&>nohup.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_perturb_norm_length=5.0 --cs_model_dir=output --cs_gpu=0"&>nohup.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_perturb_norm_length=10.0 --cs_model_dir=output1 --cs_gpu=1"&>nohup1.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_perturb_norm_length=6.0 --cs_model_dir=output2 --cs_gpu=2"&>nohup2.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_perturb_norm_length=4.0 --cs_model_dir=output3 --cs_gpu=3"&>nohup3.out&

nohup sh -c "python3 train.py --cs_tfm_ft_embed=True --cs_tfm_ft_enc_layers=12 --cs_perturb_norm_length=5.0 --cs_model_dir=output --cs_gpu=0 --cs_adv_train=False"&>nohup.out&
nohup sh -c "python3 train.py --cs_tfm_ft_embed=True --cs_tfm_ft_enc_layers=12 --cs_perturb_norm_length=5.0 --cs_model_dir=output1 --cs_gpu=1 --cs_adv_train=True"&>nohup1.out&
nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=12 --cs_perturb_norm_length=5.0 --cs_model_dir=output2 --cs_gpu=2 --cs_adv_train=True"&>nohup2.out&
nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=12 --cs_perturb_norm_length=5.0 --cs_model_dir=output2 --cs_gpu=2 --cs_adv_train=True"&>nohup2.out&

nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=10 --cs_l2_reg_coeff=0 --cs_perturb_norm_length=5.0 --cs_lambda=0.5 --cs_model_dir=output --cs_gpu=0 --cs_adv_train=False"&>nohup.out&
nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=10 --cs_l2_reg_coeff=0 --cs_perturb_norm_length=3.0 --cs_lambda=0.7 --cs_model_dir=output1 --cs_gpu=1 --cs_adv_train=True"&>nohup1.out&
nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=10 --cs_l2_reg_coeff=0 --cs_perturb_norm_length=3.0 --cs_lambda=0.3 --cs_model_dir=output2 --cs_gpu=2 --cs_adv_train=True"&>nohup2.out&
python3 train.py --cs_model_dir=output --cs_gpu=0 --cs_k_fold=1
python3 train.py --cs_model_dir=output --cs_gpu=0 --cs_train_steps=1

nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=10 --cs_l2_reg_coeff=0 --cs_model_dir=output --cs_gpu=0 --cs_adv_train=False"&>nohup.out&
nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=10 --cs_l2_reg_coeff=0 --cs_perturb_norm_length=3.0 --cs_lambda=0.3 --cs_model_dir=output1 --cs_gpu=1 --cs_adv_train=True"&>nohup1.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output --cs_gpu=0 --cs_perturb_id=0"&>nohup.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output1 --cs_gpu=1 --cs_perturb_id=1"&>nohup1.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output2 --cs_gpu=2 --cs_perturb_id=2"&>nohup2.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output3 --cs_gpu=3 --cs_perturb_id=3"&>nohup3.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output4 --cs_gpu=0 --cs_perturb_id=4"&>nohup4.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output5 --cs_gpu=1 --cs_perturb_id=5"&>nohup5.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output5 --cs_gpu=1 --cs_perturb_id=5 --cs_lambda=0.1"&>nohup5.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output6 --cs_gpu=2 --cs_perturb_id=5 --cs_lambda=0.2"&>nohup6.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output6 --cs_gpu=2 --cs_perturb_id=6"&>nohup6.out&

nohup sh -c "python3 train.py --cs_model_dir=output --cs_gpu=0"&>nohupbaseline.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output1 --cs_gpu=1 --cs_perturb_id=1"&>nohup1.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output2 --cs_gpu=2 --cs_perturb_id=2"&>nohup2.out&

nohup sh -c "python3 train.py --cs_use_clef_data=True --cs_k_fold=1 --cs_model_dir=output --cs_weight_classes_loss=True --cs_gpu=0"&>nohup.out&
nohup sh -c "python3 train.py --cs_use_clef_data=True --cs_k_fold=1 --cs_model_dir=output1 --cs_weight_classes_loss=True --cs_gpu=1 --cs_temp_adj_flag=True"&>nohup1.out&
nohup sh -c "python3 train.py --cs_use_clef_data=True --cs_k_fold=1 --cs_model_dir=output2 --cs_weight_classes_loss=True --cs_gpu=2 --cs_tfm_ft_enc_layers=2"&>nohup2.out&
nohup sh -c "python3 train.py --cs_use_clef_data=True --cs_k_fold=1 --cs_model_dir=output2 --cs_weight_classes_loss=True --cs_gpu=2 --cs_tfm_ft_enc_layers=2 --cs_restore_and_continue=True"&>nohup2c.out&

# -------------- asdf --------------

nohup sh -c "python3 train.py --cs_use_clef_data=True --cs_k_fold=1 --cs_model_dir=output2 --cs_weight_classes_loss=True --cs_gpu=0 --cs_tfm_ft_enc_layers=2 --cs_adv_train=True --cs_lambda=0.1 --cs_perturb_norm_length=3.0"&>nohup2.out&
python3 clef.py --cs_model_dir=output2/fold_01_028 --cs_tfm_ft_enc_layers=2 --cs_gpu=3

# 12 or 18

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output5 --cs_gpu=1 --cs_perturb_id=0 --cs_lambda=0.1 --cs_tfm_ft_enc_layers=2"&>nohup5.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output6 --cs_gpu=2 --cs_perturb_id=0 --cs_lambda=0.2 --cs_tfm_ft_enc_layers=2"&>nohup6.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output7 --cs_gpu=1 --cs_perturb_id=0 --cs_lambda=0.1 --cs_tfm_ft_enc_layers=2 --cs_perturb_norm_length=1.0"&>nohup7.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output8 --cs_gpu=2 --cs_perturb_id=0 --cs_lambda=0.1 --cs_tfm_ft_enc_layers=2 --cs_perturb_norm_length=2.0"&>nohup8.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=output9 --cs_gpu=2 --cs_perturb_id=0 --cs_lambda=0.1 --cs_tfm_ft_enc_layers=2 --cs_perturb_norm_length=3.0"&>nohup9.out&

nohup sh -c "python3 train.py --cs_gpu=1 --cs_model_dir=output1 --cs_tfm_ft_enc_layers=2 --cs_train_steps=5"&>nohupbaseline1.out&

nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=outputFinal --cs_gpu=1 --cs_perturb_id=5 --cs_perturb_norm_length=1.0"&>nohupFinal.out&
nohup sh -c "python3 train.py --cs_adv_train=True --cs_model_dir=outputFinal2 --cs_gpu=2 --cs_perturb_id=5 --cs_perturb_norm_length=2.0"&>nohupFinal2.out&

nohup sh -c "python3 train.py --cs_k_fold=1 --cs_adv_train=True --cs_model_dir=output --cs_gpu=0 --cs_perturb_id=5 --cs_reg_train_file=kfold_25ncs.json --cs_reg_test_file=../deprecated/disjoint_2000.json --cs_refresh_data=True --cs_train_steps=30"&>nohup.out&




# newnew new

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output --cs_gpu=0"&>nohup.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output1 --cs_gpu=1 --cs_adv_train=True"&>nohup1.out&

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output1 --cs_gpu=1 --cs_pool_strat=mean"&>nohup1.out&