python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output/032 > asdflarge.txt


python3 eval.py --gpu=1 --bert_model_size=large_wwm --cb_model_dir=/adversarial-claimspotting/output/032 > asdflarge.txt
python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-0/028 > asdf0.txt










nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-1/028"&>asdf1.txt&
nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-2/020"&>asdf2.txt&
nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-3/027"&>asdf3.txt&
nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-4/039"&>asdf4.txt&
nohup sh -c "python3 eval.py --gpu=1 --cb_model_dir=/adversarial-claimspotting/output-adv-6/023"&>asdf6.txt&