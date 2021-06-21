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

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output --cs_gpu=0 --cs_tfm_type=distilbert-base-uncased --cs_adv_train=True"&>nohup.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output1 --cs_gpu=1 --cs_pool_strat=mean"&>nohup1.out&

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output --cs_gpu=0 --cs_tfm_type=roberta-base --cs_adv_train=True"&>nohup.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output1 --cs_gpu=1 --cs_tfm_type=albert-base-v2 --cs_adv_train=True --cs_tfm_ft_enc_layers=12"&>nohup1.out&

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output2 --cs_gpu=0 --cs_adv_train=True"&>nohup2.out&


# --- BELOW ARE 2021 TESTS ---

# base

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_roberta --cs_gpu=0 --cs_tfm_type=roberta-base --cs_train_steps=30 --cs_tfm_ft_enc_layers=4"&>nohup_roberta.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_roberta_adv --cs_gpu=0 --cs_tfm_type=roberta-base --cs_train_steps=35 --cs_tfm_ft_enc_layers=6 --cs_adv_train=True"&>nohup_roberta_adv.out&

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_distilbert --cs_gpu=1 --cs_tfm_type=distilbert-base-uncased"&>nohup_distilbert.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_distilbert_adv --cs_gpu=1 --cs_tfm_type=distilbert-base-uncased --cs_adv_train=True --cs_train_steps=10"&>nohup_distilbert_adv.out&

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_albert --cs_gpu=1 --cs_tfm_type=albert-base-v2 --cs_tfm_ft_enc_layers=12 --cs_train_steps=30 --cs_lr=1e-4"&>nohup_albert.out&

nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert --cs_gpu=0 --cs_tfm_type=bert-base-uncased --cs_train_steps=15"&>nohup_bert.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert_adv --cs_gpu=0 --cs_tfm_type=bert-base-uncased --cs_train_steps=15 --cs_perturb_id=6 --cs_adv_train=True"&>nohup_bert_adv.out&

# large
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_roberta_large --cs_gpu=1 --cs_tfm_type=roberta-large --cs_train_steps=30 --cs_tfm_ft_enc_layers=4 --cs_refresh_data=True"&>nohup_roberta_large.out&


# prod
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_distilbert_adv --cs_gpu=0 --cs_tfm_type=distilbert-base-uncased --cs_adv_train=True --cs_train_steps=10 --cs_k_fold=1 --cs_reg_train_file=kfold_25ncs.json"&>nohup_distilbert_adv.out&

# clef eval
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_distilbert_adv_clef --cs_gpu=1 --cs_tfm_type=distilbert-base-uncased --cs_adv_train=True --cs_train_steps=10 --cs_use_clef_data=True --cs_k_fold=1 --"&>nohup_distilbert_adv_clef.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert_adv_clef --cs_gpu=1 --cs_tfm_type=bert-base-uncased --cs_adv_train=True --cs_train_steps=20 --cs_use_clef_data=True --cs_k_fold=1 --cs_refresh_data=True"&>nohup_bert_adv_clef.out&
nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_roberta_adv_clef --cs_gpu=1 --cs_tfm_type=roberta-base --cs_adv_train=True --cs_train_steps=20 --cs_use_clef_data=True --cs_k_fold=1 --cs_refresh_data=True"&>nohup_roberta_adv_clef.out&

python3 -m adv_transformer.benchmark --cs_model_dir=output_distilbert_adv_clef --cs_gpu=0 --cs_tfm_type=distilbert-base-uncased
python3 -m adv_transformer.benchmark --cs_gpu=1 --cs_tfm_type=bert-base-uncased
python3 -m adv_transformer.benchmark --cs_gpu=1 --cs_tfm_type=roberta-base

python3 -m adv_transformer.clef_eval_2020 --cs_model_dir=output_roberta_adv_clef --cs_gpu=1 --cs_tfm_type=roberta-base
python3 -m adv_transformer.clef_eval_2020 --cs_model_dir=output_distilbert_adv_clef --cs_gpu=1 --cs_tfm_type=distilbert-base-uncased
python3 scorer/main.py --gold_file_path="./test-input/test-input/test-gold.tsv" --pred_file_path="../clef-out.tsv"
python3 scorer/main.py --gold_file_path="./test-input/test-gold/20190619_Trump_Campain_Florida.tsv,./test-input/test-gold/20180712_Trump_NATO.tsv,./test-input/test-gold/20180731_Trump_Tampa.tsv,./test-input/test-gold/20180612_Trump_Singapore.tsv,./test-input/test-gold/20190304_Trump_CPAC.tsv,./test-input/test-gold/20180426_Trump_Fox_Friends.tsv,./test-input/test-gold/20160309_democrats_miami.tsv,./test-input/test-gold/20190731_democratic_debate_Detroit_2.tsv,./test-input/test-gold/20180628_Trump_NorthDakota.tsv,./test-input/test-gold/20170713_Trump_Roberston_interiew.tsv,./test-input/test-gold/20170512_Trump_NBC_holt_interview.tsv,./test-input/test-gold/20180615_Trump_lawn.tsv,./test-input/test-gold/20190912_democratic_debate.tsv,./test-input/test-gold/20160303_GOP_michigan.tsv,./test-input/test-gold/20181102_Trump_Huntington.tsv,./test-input/test-gold/20180821_Trump_Charleston.tsv,./test-input/test-gold/20170404_Trump_CEO_TownHall.tsv,./test-input/test-gold/20160907_NBC_commander_in_chief_forum.tsv,./test-input/test-gold/20170207_Sanders_Cruz_healthcare_debate.tsv,./test-input/test-gold/20190730_democratic_debate_Detroit_1.tsv" --pred_file_path="../clef2020_task5_20190619_Trump_Campain_Florida.tsv,../clef2020_task5_20180712_Trump_NATO.tsv,../clef2020_task5_20180731_Trump_Tampa.tsv,../clef2020_task5_20180612_Trump_Singapore.tsv,../clef2020_task5_20190304_Trump_CPAC.tsv,../clef2020_task5_20180426_Trump_Fox_Friends.tsv,../clef2020_task5_20160309_democrats_miami.tsv,../clef2020_task5_20190731_democratic_debate_Detroit_2.tsv,../clef2020_task5_20180628_Trump_NorthDakota.tsv,../clef2020_task5_20170713_Trump_Roberston_interiew.tsv,../clef2020_task5_20170512_Trump_NBC_holt_interview.tsv,../clef2020_task5_20180615_Trump_lawn.tsv,../clef2020_task5_20190912_democratic_debate.tsv,../clef2020_task5_20160303_GOP_michigan.tsv,../clef2020_task5_20181102_Trump_Huntington.tsv,../clef2020_task5_20180821_Trump_Charleston.tsv,../clef2020_task5_20170404_Trump_CEO_TownHall.tsv,../clef2020_task5_20160907_NBC_commander_in_chief_forum.tsv,../clef2020_task5_20170207_Sanders_Cruz_healthcare_debate.tsv,../clef2020_task5_20190730_democratic_debate_Detroit_1.tsv"

python3 -m adv_transformer.train --cs_model_dir=output_bert_clef --cs_gpu=0 --cs_tfm_type=bert-base-uncased --cs_train_steps=30 --cs_use_clef_data=True --cs_k_fold=1