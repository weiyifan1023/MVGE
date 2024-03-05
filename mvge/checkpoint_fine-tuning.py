import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel, ElectraModel, AutoModel
from reg_hnt.reghnt.modeling_reghnt import RegHNTModel
from reg_hnt.reghnt.model import RegHNTFineTuningModel
from reg_hnt.data.tatqa_batch_gen import TaTQABatchGen
from reg_hnt.data.data_util import OPERATOR_CLASSES_
from reg_hnt.reghnt.Vocabulary import *
from reg_hnt.reghnt.util import create_logger, set_environment
from pprint import pprint

checkpoint = torch.load('../try/checkpoint_best.ot')  # 加载断点

args = checkpoint["config"]
start_epoch = checkpoint["epoch"] + 1


train_itr = TaTQABatchGen(args, data_mode="train", encoder=args.encoder)
dev_itr = TaTQABatchGen(args, data_mode="dev", encoder=args.encoder, flag=1)
# ft_dev_itr = TaTQABatchGen(args, data_mode="dev", encoder=args.encoder)

num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
operators = OPERATOR_CLASSES_
arithmetic_op_index = [3, 4, 6, 7, 8, 9]
bert_model = RobertaModel.from_pretrained(os.path.join("../", args.roberta_model))
network = RegHNTModel(
        encoder=bert_model,
        config=bert_model.config,
        bsz=args.batch_size,
        operator_classes=len(OPERATOR_CLASSES),  # OPERATOR_CLASSES = {"SPAN": 0, "MULTI_SPAN": 1, "COUNT": 2, "ARITHMETIC": 3}
        scale_classes=len(SCALE),  # ["", "thousand", "million", "billion", "percent"]
        operator_criterion=nn.CrossEntropyLoss(),
        scale_criterion=nn.CrossEntropyLoss(),
        arithmetic_op_index=arithmetic_op_index,
        op_mode=args.op_mode,
        ablation_mode=args.ablation_mode,
    )
network.load_state_dict(torch.load('../try/checkpoint_best.pt'))
model = RegHNTFineTuningModel(args, network, num_train_steps=num_train_steps)
model.optimizer.load_state_dict(torch.load('../try/checkpoint_best.ot')["optimizer"])

logger = create_logger("RegHNT Fine-Turing", log_file=os.path.join("../try", args.log_file))

pprint(args)
set_environment(args.seed, args.cuda)

best_result = 0.79
patience_counter = 0
patience = 10
for epoch in range(start_epoch, start_epoch + 50):
    model.reset()
    logger.info('At epoch {}'.format(epoch))
    for step, batch in enumerate(train_itr):  # __iter__ 生成器generator
        model.update(batch)
        if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
            logger.info("Updates[{0:6}] train loss[{1:.5f}] .\r\n".format(model.updates, model.train_loss.avg))
            model.avg_reset()
    metrics = model.get_metrics(logger)  # train data
    model.avg_reset()  # loss.reset()
    model.reset()  # _metrics.reset()
    # dev data
    model.evaluate(dev_itr)
    metrics = model.get_metrics(logger)
    model.avg_reset()
    if metrics["f1"] > best_result:
        # Save the model at each epoch.
        save_prefix = os.path.join("../try", "checkpoint_fine-tune_{}".format(epoch - start_epoch))  # fine-tune 轮数
        model.save(save_prefix, epoch + 1)
        best_result = metrics["f1"]
        patience_counter = 0
        logger.info("Best eval F1 {} at epoch {}.\r\n".format(best_result, epoch))
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("-> Early stopping: patience limit reached, stopping...")
        break




print("ok")









