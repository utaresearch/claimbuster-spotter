nohup sh -c "python3 train.py --cb_model_dir=output_reg --gpu=0 --adv_train=False --tfm_ft_embed=True --tfm_ft_enc_layers=12"&>nohup.out&
nohup sh -c "python3 train.py --cb_model_dir=output_adv --gpu=1 --adv_train=True"&>nohup1.out&
nohup sh -c "python3 train.py --cb_model_dir=output_reg_large --gpu=2 --bert_model_size=large_wwm --adv_train=False"&>nohup2.out&

nohup sh -c "python3 train.py --cb_model_dir=output_reg_1 --gpu=1 --adv_train=False"&>nohup1.out&
nohup sh -c "python3 train.py --cb_model_dir=output_adv_2 --gpu=2 --adv_train=True"&>nohup2.out&

nohup sh -c "python3 train.py --cb_model_dir=output --gpu=0 --l2_reg_coeff=0"&>nohup.out&

nohup sh -c "python3 train.py --tfm_ft_enc_layers=3 --l2_reg_coeff=0 --restore_and_continue=True"&>nohup.out&
nohup sh -c "python3 train.py --cb_model_dir=output1 --restore_and_continue=True --gpu=1 --lr=2e-5 --tfm_ft_enc_layers=3 --l2_reg_coeff=0"&>nohup1.out&


nohup sh -c "python3 train.py --bert_model_size=large_wwm --lr=1e-4"&>nohup.out&