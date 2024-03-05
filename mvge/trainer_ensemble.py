import io, requests, zipfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import json
import argparse
from datetime import datetime
from reg_hnt import options
import torch.nn as nn
from pprint import pprint
from reg_hnt.data.data_util import OPERATOR_CLASSES_
from reg_hnt.reghnt.util import create_logger, set_environment
from reg_hnt.reghnt.Vocabulary import *
from reg_hnt.data.tatqa_batch_gen import TaTQABatchGen
from transformers import RobertaModel, BertModel, ElectraModel, AutoModel
from reg_hnt.reghnt.modeling_reghnt_ensemble import RegHNTModel
from reg_hnt.reghnt.model import RegHNTFineTuningModel
from pathlib import Path
import torch
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("RegHNT training task.")
options.add_data_args(parser)
options.add_train_args(parser)
options.add_bert_args(parser)
parser.add_argument("--encoder", type=str, default='roberta_large')
parser.add_argument("--op_mode", type=int, default=0)
parser.add_argument("--ablation_mode", type=int, default=0)
parser.add_argument("--test_data_dir", type=str, default="./reg_hnt/cache")

args = parser.parse_args()

args.cuda = args.gpu_num > 0
args.batch_size = args.batch_size // args.gradient_accumulation_steps
result_time = datetime.now().strftime("%m%d%H%M")  # 训练结果保存path,日期命名形式

if args.ablation_mode != 0:
    args.save_dir = args.save_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)
    args.data_dir = args.data_dir + "_{}_{}".format(args.op_mode, args.ablation_mode)

result_saved_path = os.path.join(args.save_dir, result_time + "-MVAG-ensemble")  # 新设定path
Path(result_saved_path).mkdir(parents=True, exist_ok=True)
# Path(args.save_dir).mkdir(parents=True, exist_ok=True)
# args_path = os.path.join(args.save_dir, result_time, "args.json")
args_path = os.path.join(result_saved_path, "args.json")
with open(args_path, "w") as f:
    json.dump((vars(args)), f)

logger = create_logger("Roberta Training", log_file=os.path.join(result_saved_path, args.log_file))  # log日志存储path

pprint(args)
set_environment(args.seed, args.cuda)


def main():
    best_result, best_op_score, best_scale_score, best_arithmetic, best_count, best_multi, best_span = float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf")
    logger.info("Loading data...")


    train_itr = TaTQABatchGen(args, data_mode="train", encoder=args.encoder)
    if args.ablation_mode != 3:
        dev_itr = TaTQABatchGen(args, data_mode="dev", encoder=args.encoder)
    else:
        dev_itr = TaTQABatchGen(args, data_mode="dev", encoder=args.encoder)

    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)  # len(train_itr) == Batch == Iteration
    logger.info("Num update steps {}!".format(num_train_steps))

    logger.info(f"Build {args.encoder} model.")
    if args.encoder == 'bert':
        bert_model = BertModel.from_pretrained(args.bert_model)
    elif args.encoder == 'deberta':
        bert_model = AutoModel.from_pretrained(args.deberta_model)
    elif args.encoder == 'roberta_base':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)
    elif args.encoder == 'roberta_large':
        bert_model = RobertaModel.from_pretrained(args.roberta_model)  # 改 服务器版本
        # bert_model = RobertaModel.from_pretrained(os.path.join("../", args.roberta_model))

    if args.ablation_mode == 0:
        operators = OPERATOR_CLASSES_

    if args.ablation_mode == 0:
        arithmetic_op_index = [3, 4, 6, 7, 8, 9]

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
    logger.info("Build optimizer etc...")

    model = RegHNTFineTuningModel(args, network, num_train_steps=num_train_steps)

    # Data for loss curves plot.
    patience = 20
    patience_counter = 0
    epochs_count = []
    train_losses, train_f1 = [], []
    valid_losses, valid_f1 = [], []

    train_start = datetime.now()  # 训练开始时间
    first = True
    for epoch in range(1, args.max_epoch + 1):
        epochs_count.append(epoch)
        model.reset()
        if not first:  # 每个epoch shuffle一次
            train_itr.reset()  # shuffle操作
        first = False
        logger.info('At epoch {}'.format(epoch))
        train_loss = 0
        num = 0
        for step, batch in enumerate(train_itr):  # __iter__ 生成器generator
            model.update(batch)  # 反向传播，得到正常的grad
            if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                logger.info("Updates[{0:6}] train loss[{1:.5f}] remaining[{2}].\r\n".format(model.updates, model.train_loss.avg,
                    str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))  # 当前时间-开始时间
                train_loss += model.train_loss.avg  # 存储当前epoch 每个 step 的loss
                num += 1  # 几个loss
                model.avg_reset()

        model.get_metrics(logger)  # train data
        model.reset()  # _metrics.reset()
        model.avg_reset()  # loss.reset()
        # dev data
        # model.evaluate(dev_itr)
        model.predict(dev_itr)  # ensemble 不用loss
        metrics = model.get_metrics(logger)
        model.avg_reset()
        # if metrics["f1"] > best_result:
        #     save_prefix = os.path.join(args.save_dir, result_time, "checkpoint_best")  # model_save path: try/
        #     model.save(save_prefix, epoch)
        #     best_result = metrics["f1"]
        #     patience_counter = 0
        #     logger.info("Best eval F1 {} at epoch {}.\r\n".format(best_result, epoch))
        #
        #     # Plotting of the loss curves for the train and validation sets.
        #     draw_loss_f1(epochs_count, train_losses, valid_losses, train_f1, valid_f1)
        # else:
        #     patience_counter += 1
        #
        # if patience_counter >= patience:
        #     print("-> Early stopping: patience limit reached, stopping...")
        #     break

        # ensemble
        # result_time = result_time + "-ensemble"
        if metrics["em"] > best_result:
            save_prefix = os.path.join(result_saved_path, "best_em_checkpoint")
            model.save(save_prefix, epoch)
            best_result = metrics["em"]
            logger.info("Best eval EM {} at epoch {}.\r\n".format(best_result, epoch))
        if metrics["op"] > best_op_score:
            save_prefix = os.path.join(result_saved_path, "best_op_checkpoint")
            model.save(save_prefix, epoch)
            best_op_score = metrics["op"]
            logger.info("Best eval OP {} at epoch {}.\r\n".format(best_op_score, epoch))
        if metrics["scale"] > best_scale_score:
            save_prefix = os.path.join(result_saved_path, "best_scale_checkpoint")
            model.save(save_prefix, epoch)
            best_scale_score = metrics["scale"]
            logger.info("Best eval SCALE {} at epoch {}.\r\n".format(best_scale_score, epoch))
        if metrics["arithmetic_em"] > best_arithmetic:
            save_prefix = os.path.join(result_saved_path, "best_arithmetic_checkpoint")
            model.save(save_prefix, epoch)
            best_arithmetic = metrics["arithmetic_em"]
            logger.info("Best eval ARITHMETIC {} at epoch {}.\r\n".format(best_arithmetic, epoch))
        if metrics["count_em"] > best_count:
            save_prefix = os.path.join(result_saved_path, "best_count_checkpoint")
            model.save(save_prefix, epoch)
            best_count = metrics["count_em"]
            logger.info("Best eval COUNT {} at epoch {}.\r\n".format(best_count, epoch))
        if metrics["multi_em"] > best_multi:
            save_prefix = os.path.join(result_saved_path, "best_multi_checkpoint")
            model.save(save_prefix, epoch)
            best_multi = metrics["multi_em"]
            logger.info("Best eval MULTI {} at epoch {}.\r\n".format(best_multi, epoch))
        if metrics["span_em"] > best_span:
            save_prefix = os.path.join(result_saved_path, "best_span_checkpoint")
            model.save(save_prefix, epoch)
            best_span = metrics["span_em"]
            logger.info("Best eval SPAN {} at epoch {}.\r\n".format(best_span, epoch))

    # fine-tuning on dev      =====================================================
    fine_tune = 0
    if fine_tune == 1:
        logger.info("Early stopping: Start Fine-tune process")
        start_epoch = epochs_count[-1] + 1
        valid_last_loss = valid_losses[-1]
        valid_last_f1 = valid_f1[-1]
        for epoch in range(start_epoch, start_epoch + 5):
            model.reset()
            if not first:  # 每个epoch shuffle一次
                dev_itr.reset()  # shuffle操作
            logger.info('At epoch {}'.format(epoch))
            train_loss = 0
            num = 0
            for step, batch in enumerate(dev_itr):  # __iter__ 生成器generator
                model.update(batch)
                if model.step % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or model.step == 1:
                    logger.info("Updates[{0:6}] train loss[{1:.5f}] remaining[{2}].\r\n"
                                .format(model.updates, model.train_loss.avg, str((datetime.now() - train_start) / (step + 1) * (num_train_steps - step - 1)).split('.')[0]))  # 当前时间-开始时间
                    train_loss += model.train_loss.avg  # 存储当前epoch 每个 step 的loss
                    num += 1  # 几个loss
                    model.avg_reset()
            metrics = model.get_metrics(logger)  # train data
            # epochs_count.append(epoch)
            # train_f1.append(metrics["f1"])
            # train_losses.append(train_loss / num)
            # valid_f1.append(valid_last_f1)
            # valid_losses.append(valid_last_loss)  # 保存train data 最后的loss; f1
            model.avg_reset()  # loss.reset()
            # Save the model at each epoch.
            save_prefix = os.path.join(result_saved_path, "checkpoint_fine-tune_{}".format(epoch - start_epoch))  # fine-tune 轮数
            model.save(save_prefix, epoch)

        # Plotting of the loss curves for the train and validation sets.
        # draw_loss_f1(epochs_count, train_losses, valid_losses, train_f1, valid_f1)



def draw_loss_f1(epochs_count, train_losses, valid_losses, train_f1, valid_f1):
    plt.figure()
    # loss
    # 数据缩放 (under,up)归一化到 (under,under+space)
    space = 5  # 间距
    mini = 0  # 坐标轴下界
    up = 200  # 缩放上界
    under = 30  # 缩放下界

    train_losses = np.array(train_losses)
    index = np.where(train_losses[train_losses > under])
    train_losses[index] = (train_losses[index] - under) / (up - under) * space + under  # 归一化到[under,under+space]
    train_losses = train_losses.tolist()
    num = (under - mini) / space
    # 原先坐标
    y_scale = [mini + i * space for i in range(int(num + 3))]  # 前1后2
    # 更改后坐标
    y_scale2 = []
    for i in range(len(y_scale) - 1):
        y_scale2.append(str(y_scale[i]))
    y_scale2.append(str(up))  # 最后一个显示真坐标 up
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")  # dev不需要处理 不属于压缩区间
    plt.yticks(y_scale, y_scale2)  # 更改刻度
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    plt.savefig(os.path.join(result_saved_path, "loss.png"))
    plt.close()

    # F1 数据缩放 (under,up)归一化到 (up-space,up)
    maxi = 1.0  # 坐标轴上界
    up = 0.5  # 缩放上界
    under = 0  # 缩放下界
    space = 0.1  # 间距
    train_f1, valid_f1 = np.array(train_f1), np.array(valid_f1)
    # train
    index = np.where(train_f1[train_f1 < up])
    train_f1[index] = (train_f1[index] - under) / (up - under) * space + (up - space)  # 归一化到[up-space,up]
    # valid
    index = np.where(valid_f1[valid_f1 < up])
    valid_f1[index] = (valid_f1[index] - under) / (up - under) * space + (up - space)  # 归一化到[up-space,up]
    train_f1, valid_f1 = train_f1.tolist(), valid_f1.tolist()

    num = (maxi - up) / space
    # 原先坐标
    y_scale = [round(up - space + i * space, 1) for i in range(int(num + 2))]
    # 更改后坐标
    y_scale2 = []
    y_scale2.append(str(under))  # 第一个显示坐标替换为 under
    for i in range(1, len(y_scale)):
        y_scale2.append(str(y_scale[i]))
    plt.plot(epochs_count, train_f1, "-r")
    plt.plot(epochs_count, valid_f1, "-b")
    plt.yticks(y_scale, y_scale2)  # 更改刻度
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.legend(["Training F1", "Validation F1"])
    plt.title("Model F1")
    plt.savefig(os.path.join(result_saved_path, "f1.png"))
    plt.close()

if __name__ == "__main__":
    main()
    print("MVG wo FDG ensemble on the 2")
    print("MVAG  ensemble on the 3 ")
    print("MVAG  ensemble on the 7 Relation-view")
    print("MVAG  ensemble on the 8 Tabula-view")
    print("MVAG  ensemble on the 9 Numerical-view")





