for i in {1..3}
do
#  CUDA_VISIBLE_DEVICES=$i nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert_$i --cs_tfm_type=bert-base-uncased --cs_train_steps=10 --cs_perturb_id=6 --cs_adv_train=True"&>nohup_bert_$i.out&
#  CUDA_VISIBLE_DEVICES=$i nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_distilbert_$i --cs_tfm_type=distilbert-base-uncased --cs_train_steps=7 --cs_adv_train=True"&>nohup_distilbert_$i.out&
CUDA_VISIBLE_DEVICES=$i nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_roberta_$i --cs_tfm_type=roberta-base --cs_train_steps=30 --cs_tfm_ft_enc_layers=6"&>nohup_roberta_$i.out&
done