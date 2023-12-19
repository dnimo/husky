import sys

sys.path.append('/home/jovyan/mi-drive/medinfo_lab/Research_Projects/zhang/Husky/tools/')

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from Model.data.mlm_dataset import *
from Model.layers.Roberta_mlm import RobertaMlm
from datasets import load_dataset, Dataset
import logging
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


def chunk_examples(examples):
    chunks = []
    for sentence in examples['text']:
      if len(sentence) < 512:
        chunks += [sentence]
      else:
        chunks += [sentence[i:i+512] for i in range(0, len(sentence) - 512 + 1, 512)]
    return {"chunks": chunks}

def train_ddp_accelerate():
  accelerator = Accelerator()
  device = accelerator.device
  
  if Debug:
    print('开始训练 %s' % get_time())
  train_dataset = load_dataset("text", data_files=CorpusPath)
  train_val = train_dataset["train"].train_test_split(
    test_size=0.8,
    shuffle=True,
    seed=42
  )
  train_data = train_val["train"].map(chunk_examples, batched=True, remove_columns=["text"])
  train_data = RobertaDataSet(train_data["chunks"][:-1])
  dataloader = DataLoader(dataset=train_data, batch_size=BatchSize, shuffle=True, drop_last=True)
  print("loaded data.")
  
  roberta = RobertaMlm().to(device)
  roberta = accelerator.prepare(roberta)
  optim = Adam(roberta.parameters(), lr=MLMLearningRate)
  criterion = nn.CrossEntropyLoss()
  dataloader, optim = accelerator.prepare(dataloader, optim)

  if UsePretrain and os.path.exists(PretrainPath):
      if SentenceLength == 512:
          print('开始加载预训练模型！')
          roberta.load_pretrain(SentenceLength)
          print('完成加载预训练模型！')
      else:
          print('开始加载本地模型！')
          roberta.load_pretrain(SentenceLength)
          print('完成加载本地模型！')

  for epoch in range(MLMEpochs):
      # train
      roberta.train()
      data_iter = tqdm(enumerate(dataloader), desc='EP_%s:%d' % ('train', epoch), total=len(dataloader))
      print_loss = 0.0
      with accelerator.accumulate(roberta):
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            segment_ids = data['segment_ids']
            label = data['token_ids_labels']
            
            mlm_output = roberta(input_token, segment_ids).permute(0, 2, 1)
            
            mask_loss = criterion(mlm_output, label)
            print_loss = mask_loss.item()
            accelerator.backward(mask_loss)
            optim.step()
            optim.zero_grad()
        print('EP_%d mask loss:%s' % (epoch, print_loss))

      # save
      output_path =  "kuhpBERT" + '.ep%d' % epoch
      accelerator.wait_for_everyone()
      if accelerator.is_local_main_process:
        unwrap_model = accelerator.unwrap_model(roberta)
        unwrap_optim = accelerator.unwrap_model(optim)
        torch.save({
        'model_state' : unwrap_model.state_dict(),
        'optim_state' : unwrap_optim.state_dict()}, output_path + f'ckpt_{epoch+1}.pt')
      logger.info(f'checkpoint ckpt_{epoch+1}.pt is saved...')
      
  
if __name__ == '__main__':
   train_ddp_accelerate()

#   train_val = train_dataset["train"].train_test_split(
#     test_size=1000,
#     shuffle=True,
#     seed=42
# )


# test_data = train_val["test"]

# train_data = train_data.map(padding_cutting_examples, batched=True, remove_columns=train_data.column_names)
# test_data = test_data.map(chunk_examples, batched=True, remove_columns=test_data.column_names)

# train_data = train_data.map(lambda x: RobertaDataSet(x))
# testset = RobertaTestSet(test_data['chunks'])



# training_args = TrainingArguments(
#     "basic-trainer",
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=64,
#     num_train_epochs=1,
#     evaluation_strategy="epoch",
#     remove_unused_columns=False
# )

# class MyTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(inputs["x"])
#         target = inputs["labels"]
#         loss = F.nll_loss(outputs, target)
#         return (loss, outputs) if return_outputs else loss

# trainer = MyTrainer(
#     model,
#     training_args,
#     train_dataset=train_dset,
#     eval_dataset=test_dset,
#     data_collator=collate_fn,
# )


  #     # test
  #     with torch.no_grad():
  #         roberta.eval()
  #         test_count = 0
  #         top1_count = 0
  #         top5_count = 0
  #         for test_data in testset:
  #             input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
  #             segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
  #             input_token_list = input_token.tolist()
  #             input_len = len([x for x in input_token_list[0] if x]) - 2
  #             label_list = test_data['token_ids_labels'].tolist()
  #             mlm_output = roberta(input_token, segment_ids)[:, 1:input_len + 1, :]
  #             output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
  #             output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()

  #             # 累计数值
  #             test_count += input_len
  #             for i in range(input_len):
  #                 label = label_list[i + 1]
  #                 if label == output_topk[i][0]:
  #                     top1_count += 1
  #                 if label in output_topk[i]:
  #                     top5_count += 1

  #         if test_count:
  #             top1_acc = float(top1_count) / float(test_count)
  #             acc = round(top1_acc, 2)
  #             print('top1纠正正确率：%s' % acc)
  #             top5_acc = float(top5_count) / float(test_count)
  #             acc = round(top5_acc, 2)
  #             print('top5纠正正确率：%s\n' % acc)