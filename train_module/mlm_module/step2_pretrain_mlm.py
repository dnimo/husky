import sys

sys.path.append('/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/tools/')

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from Model.base.RoBERTa.data.mlm_dataset import *
from Model.base.RoBERTa.layers.Roberta_mlm import RobertaMlm
from datasets import load_dataset, Dataset


window_size = 510
def chunk_examples(examples):
    chunks = []
    for sentence in examples['summary']:
        chunks += [sentence[i:i+window_size] for i in range(0, len(sentence) - window_size + 1, 510)]
    return {"chunks": chunks}


if Debug:
    print('开始训练 %s' % get_time())
onehot_type = False
roberta = RobertaMlm().to(device)
if Debug:
    print('Total Parameters:', sum([p.nelement() for p in roberta.parameters()]))

if UsePretrain and os.path.exists(PretrainPath):
    if SentenceLength == 512:
        print('开始加载预训练模型！')
        roberta.load_pretrain(SentenceLength)
        print('完成加载预训练模型！')
    else:
        print('开始加载本地模型！')
        roberta.load_pretrain(SentenceLength)
        print('完成加载本地模型！')
        

train_dataset = load_dataset("json", data_files=CorpusPath)

train_val = train_dataset["train"].train_test_split(
    test_size=1000,
    shuffle=True,
    seed=42
)

train_data = train_val["train"]
# test_data = train_val["test"]
  
train_data = train_data.map(chunk_examples, batched=True, remove_columns=train_data.column_names)
# test_data = test_data.map(chunk_examples, batched=True, remove_columns=test_data.column_names)


train_data = RobertaDataSet(train_data['chunks'][:-1], onehot_type)
# testset = RobertaTestSet(test_data['chunks'])


dataloader = DataLoader(dataset=train_data, batch_size=BatchSize, shuffle=True, drop_last=True)

optim = Adam(roberta.parameters(), lr=MLMLearningRate)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(MLMEpochs):
    # train
    if Debug:
        print('第%s个Epoch %s' % (epoch, get_time()))
    roberta.train()
    data_iter = tqdm(enumerate(dataloader),
                        desc='EP_%s:%d' % ('train', epoch),
                        total=len(dataloader),
                        bar_format='{l_bar}{r_bar}')
    print_loss = 0.0
    for i, data in data_iter:
        if Debug:
            print('生成数据 %s' % get_time())
        data = {k: v.to(device) for k, v in data.items()}
        input_token = data['input_token_ids']
        segment_ids = data['segment_ids']
        label = data['token_ids_labels']
        if Debug:
            print('获取数据 %s' % get_time())
        mlm_output = roberta(input_token, segment_ids).permute(0, 2, 1)
        if Debug:
            print('完成前向 %s' % get_time())
        mask_loss = criterion(mlm_output, label)
        print_loss = mask_loss.item()
        optim.zero_grad()
        mask_loss.backward()
        optim.step()
        if Debug:
            print('完成反向 %s\n' % get_time())

    print('EP_%d mask loss:%s' % (epoch, print_loss))

    # save
    output_path = FinetunePath + '.ep%d' % epoch
    torch.save(roberta.cpu(), output_path)
    roberta.to(device)
    print('EP:%d Model Saved on:%s' % (epoch, output_path))