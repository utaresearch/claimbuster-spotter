for i in {0..3}
do
#  CUDA_VISIBLE_DEVICES=$i nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert_$i --cs_tfm_type=bert-base-uncased --cs_train_steps=10 --cs_perturb_id=6 --cs_adv_train=True"&>nohup_bert_$i.out&
  CUDA_VISIBLE_DEVICES=$i nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_distilbert_$i --cs_tfm_type=distilbert-base-uncased --cs_train_steps=10 --cs_adv_train=True --cs_perturb_id=6"&>nohup_distilbert_$i.out&
done