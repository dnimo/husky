import time
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4'

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

VocabPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/kuhpTokenizer/40000_vocab/bert_bpe_vocab_40000.vocab'

TokenizerPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/kuhpTokenizer/40000_vocab/bert_bpe_vocab_40000.model'

# ## mlm模型文件路径 ## #
CorpusPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/dataUnits/del_none_data_for_train_tokenizer.txt'
# samples = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/dataUnits/sample.txt'
PronunciationPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/tools/Model/base/RoBERTa/dataUnits/char_meta.json'

# Debug开关
Debug = False

# 使用预训练模型开关
UsePretrain = False

# 任务模式
ModelClass = 'RobertaMlm'
AttentionMask = True

# ## MLM训练调试参数开始 ## #
MLMEpochs = 2
WordGenTimes = 2
if WordGenTimes > 1:
    RanWrongDivisor = 1.0
else:
    RanWrongDivisor = 0.15
HiddenLayerNum = 12
MLMLearningRate = 1e-05
if ModelClass == 'RobertaMlm':
    BatchSize = 16
    SentenceLength = 514
    PretrainPath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/tools/Model/base/RoBERTa/checkpoint/pretrain/pytorch_model.bin'
FinetunePath = '/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/tools/Model/base/RoBERTa/checkpoint/finetune/mlm_trained_%s.model' % SentenceLength
# ## MLM训练调试参数结束 ## #

# ## MLM通用参数 ## #
DropOut = 0.1
MaskRate = 0.15
VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
HiddenSize = 768
IntermediateSize = 3072
AttentionHeadNum = 12

# 参数名配对
local2target_emb = {
    'roberta_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    'roberta_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
    'roberta_emd.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'roberta_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.weight',
    'roberta_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.bias'
}

local2target_transformer = {
    'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.weight',
    'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.bias',
    'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.weight',
    'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.bias',
}

# ## NER训练调试参数开始 ## #
NEREpochs = 8
NERLearningRate = 1e-3
NerBatchSize = 8
MedicineLength = 32
# ## NER训练调试参数结束 ## #

# ## ner模型文件路径 ## #
NerSourcePath = '../../dataUnits/src_data/ner_src_data.txt'
NerCorpusPath = '../../dataUnits/train_data/ner_train.txt'
NerTestPath = '../../dataUnits/test_data/ner_test.txt'
Class2NumFile = '../../dataUnits/train_data/c2n.pickle'
NerFinetunePath = '../../checkpoint/finetune/ner_trained_%s.model' % MedicineLength

# ## NER通用参数 ## #
NormalChar = 'ptzf'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
