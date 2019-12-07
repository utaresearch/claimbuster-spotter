python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output/032 > asdflarge.txt


python3 eval.py --gpu=1 --bert_model_size=large_wwm --cb_model_dir=/adversarial-claimspotting/output/032 > asdflarge.txt
python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-0/028 > asdf0.txt



nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-1/028"&>asdf1.txt&
nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-2/020"&>asdf2.txt&
nohup sh -c "python3 eval.py --gpu=3 --cb_model_dir=/adversarial-claimspotting/output-adv-3/027"&>asdf3.txt&
nohup sh -c "python3 eval.py --gpu=3 --cb_model_dir=/adversarial-claimspotting/output-adv-4/039"&>asdf4.txt&

nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-6/023"&>asdf6.txt&

python3 clef.py --cb_model_dir=reg_clef --use_clef_data=True --gpu=2 --num_classes=2
cp clef_out.csv ../clef2019-factchecking-task1/scorer/data/
python3 fatma_prc.py
python main.py --gold_file_path="data/g1.tsv, data/g2.tsv, data/g3.tsv, data/g4.tsv, data/g5.tsv, data/g6.tsv, data/g7.tsv" --pred_file_path="data/p1.tsv, data/p2.tsv, data/p3.tsv, data/p4.tsv, data/p5.tsv, data/p6.tsv, data/p7.tsv"


nohup sh -c "python3 pretrain.py --cb_model_dir=reg_clef --gpu=0 --use_clef_data=True --weight_classes_loss=True -num_classes=2 --restore_and_continue=True"&>nohup.out&
nohup sh -c "python3 advtrain.py --cb_model_dir=adv_clef --gpu=1 --use_clef_data=True --weight_classes_loss=True -num_classes=2 --restore_and_continue=True"&>nohup1.out&
nohup sh -c "python3 pretrain.py --cb_model_dir=reg_ours_clef --gpu=2 --use_clef_data=True --combine_ours_clef_data=True --weight_classes_loss=True -num_classes=2 --restore_and_continue=True"&>nohup2.out&
nohup sh -c "python3 advtrain.py --cb_model_dir=adv_ours_clef --gpu=3 --use_clef_data=True --combine_ours_clef_data=True --weight_classes_loss=True -num_classes=2 --restore_and_continue=True"&>nohup3.out&


python3 regtrain.py --cb_model_dir=reg_ours_clef --gpu=2 --use_clef_data=True --combine_ours_clef_data=True --weight_classes_loss=True -num_classes=2 --restore_and_continue=False