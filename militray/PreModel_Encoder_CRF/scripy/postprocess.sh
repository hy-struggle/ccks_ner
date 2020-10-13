for((fold=0;fold<5;fold++));
do
echo 'START FOLD_'$fold'...'
python ../postprocess.py \
--ex_index=13 \
--num_fold=5 \
--fold_id=$fold \
--mode=test \
--num_samples=400 \
--threshold=2
echo 'FOLD_'$fold 'done'
done