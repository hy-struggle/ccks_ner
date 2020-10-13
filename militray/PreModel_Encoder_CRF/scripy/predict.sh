for((fold=0;fold<5;fold++));
do
echo 'START FOLD_'$fold'...'
python ../predict.py \
--ex_index=1 \
--fold_id=$fold \
--device_id=3 \
--mode=test \
--pre_model_type=NEZHA \
--ds_encoder_type=TENER
echo 'FOLD_'$fold 'done'
done