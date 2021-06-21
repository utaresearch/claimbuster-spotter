for i in {0..3}
do
  nohup sh -c "python3 -m adv_transformer.train --cs_model_dir=output_bert --cs_gpu=$i --cs_tfm_type=bert-base-uncased --cs_train_steps=15"&>nohup_bert_$i.out&
done