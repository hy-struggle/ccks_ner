for((fold=0;fold<5;fold++));
do
echo 'START FOLD_'$fold'...'
python ../train.py \
--ex_index=1 \
--fold_id=$fold \
--epoch_num=10 \
--device_id=0 \
--pre_model_type=NEZHA \
--ds_encoder_type=LSTM
echo 'FOLD_'$fold 'done'
done