for i in {2..3}
do
  CUDA_VISIBLE_DEVICES=$i nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert_$i --cs_tfm_type=bert-base-uncased --cs_train_steps=15 --cs_perturb_id=6 --cs_adv_train=True"&>nohup_bert_$i.out&
done