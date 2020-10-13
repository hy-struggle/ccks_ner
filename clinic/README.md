## 文档结构

`PreModel_Encoder_CRF`：Pretrain Model + CRF范式

`PreModel_Encoder_CRF_WordAug`：Pretrain Model + CRF + 词边界增强

`PreModel_MRC`：MRC for NER

`NEZHA_MultiTask`：自适应多任务NER

## 代码结构

数据预处理：`python preprocess.py`

训练：`sh ./scripy/train.sh`

预测：`sh ./scripy/predict.sh`

后处理：`sh ./scripy/postprocess.sh`

结果融合：`python ensemble.py`