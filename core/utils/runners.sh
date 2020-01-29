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

nohup sh -c "python3 train.py --cs_tfm_ft_embed=False --cs_tfm_ft_enc_layers=10 --cs_perturb_norm_length=5.0 --cs_lambda=0.5 --cs_model_dir=output --cs_gpu=0 --cs_adv_train=True"&>nohup.out&