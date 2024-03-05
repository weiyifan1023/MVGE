import torch
import torch.nn as nn
from reg_hnt.reghnt.optimizer import BertAdam as Adam
from reg_hnt.reghnt.util import AverageMeter
from reg_hnt.reghnt.utils.adversarial_training import FGM
from tqdm import tqdm

SCALE = ["", "thousand", "million", "billion", "percent"]


class RegHNTPredictModel():
    def __init__(self, args, network):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network

        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network

        if self.args.gpu_num > 0:
            self.network.cuda()

    def avg_reset(self):
        self.train_loss.reset()
        self.dev_loss.reset()

    @torch.no_grad()
    def evaluate(self, dev_data_list, epoch=None):
        dev_data_list.reset()
        self.network.eval()
        for batch in tqdm(dev_data_list):
            output_dict = self.network(batch)   # 原先**batch, mode="eval", epoch=epoch
            loss = output_dict["loss"]
            self.dev_loss.update(loss.item(), 1)
        self.network.train()

    @torch.no_grad()
    def predict(self, test_data_list, compute=True):  # predictor.py test数据集 compute=F,只需要给出output_dict 生成json文件
        test_data_list.reset()
        self.network.eval()
        pred_json = {}
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(batch, compute)  # 原先**batch, mode="eval"
            pred_answer = output_dict["answer"]
            pred_scale = output_dict["scale"]
            question_id = output_dict["question_uid"]
            for i in range(len(question_id)):
                pred_json[question_id[i]] = [pred_answer[i], pred_scale[i]]
        return pred_json

    # @torch.no_grad()
    # def predict2(self, test_data_list):
    #     test_data_list.reset()
    #     self.network.eval()
    #     pred_json = {}
    #     for batch in tqdm(test_data_list):
    #         output_dict = self.network.predict2(batch)  # 原先**batch, mode="eval"
    #         pred_answer = output_dict["answer"]
    #         pred_scale = output_dict["scale"]
    #         question_id = output_dict["question_uid"]
    #         for i in range(len(question_id)):
    #             pred_json[question_id[i]] = [pred_answer[i], pred_scale[i]]
    #     return pred_json

    # ensemble
    @torch.no_grad()
    def predict_op(self, test_data_list, compute=False):
        test_data_list.reset()
        self.network.eval()
        pred_json = {}
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(batch, compute)
            question_id = output_dict["question_uid"]
            operator = output_dict["operator"]
            gold_op = batch['operator_class']  # gold op
            operator = gold_op.tolist()
            for i in range(len(question_id)):
                pred_json[question_id[i]] = [operator[i]]
        return pred_json

    @torch.no_grad()
    def predict_scale(self, test_data_list, scale_predict_dict, compute=False):
        test_data_list.reset()
        self.network.eval()
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(batch, compute)
            question_id = output_dict["question_uid"]
            scale = output_dict["scale"]
            gold_scale = batch['scale_class']
            scale = gold_scale.tolist()
            for i in range(len(question_id)):
                scale_predict_dict[question_id[i]].append(SCALE.index(scale[i]))
        return scale_predict_dict

    @torch.no_grad()
    def predict_span(self, test_data_list, answer_predict_dict, compute=False):
        test_data_list.reset()
        self.network.eval()
        for batch in tqdm(test_data_list):
            op_scale = answer_predict_dict[batch['question_uid'][0]][0:2]
            batch['op_scale'] = op_scale
            output_dict = self.network.predict(batch, compute)
            question_id = output_dict["question_uid"]
            pred_answer = output_dict["answer"]
            pred_scale = output_dict["scale"]
            for i in range(len(question_id)):
                if answer_predict_dict[question_id[i]][0] == 0:  # op是span
                    best_scale = SCALE[answer_predict_dict[question_id[i]][1]]  # scale_model 得到的best scale
                    if len(answer_predict_dict[question_id[i]]) == 2:
                        # answer_predict_dict[question_id[i]].append([pred_answer[i], pred_scale[i]])
                        answer_predict_dict[question_id[i]].append([pred_answer[i], best_scale])
                    else:
                        # answer_predict_dict[question_id[i]][2] = [pred_answer[i], pred_scale[i]]  # 覆盖answer和scale
                        answer_predict_dict[question_id[i]][2] = [pred_answer[i], best_scale]
        return answer_predict_dict

    @torch.no_grad()
    def predict_multi(self, test_data_list, answer_predict_dict, compute=False):
        test_data_list.reset()
        self.network.eval()
        for batch in tqdm(test_data_list):
            op_scale = answer_predict_dict[batch['question_uid'][0]][0:2]
            batch['op_scale'] = op_scale
            output_dict = self.network.predict(batch, compute)
            question_id = output_dict["question_uid"]
            pred_answer = output_dict["answer"]
            pred_scale = output_dict["scale"]
            for i in range(len(question_id)):
                if answer_predict_dict[question_id[i]][0] == 1:  # multi-span
                    best_scale = SCALE[answer_predict_dict[question_id[i]][1]]
                    if len(answer_predict_dict[question_id[i]]) == 2:
                        # answer_predict_dict[question_id[i]].append([pred_answer[i], pred_scale[i]])
                        answer_predict_dict[question_id[i]].append([pred_answer[i], best_scale])
                    else:
                        # answer_predict_dict[question_id[i]][2] = [pred_answer[i], pred_scale[i]]
                        answer_predict_dict[question_id[i]][2] = [pred_answer[i], best_scale]
        return answer_predict_dict

    @torch.no_grad()
    def predict_count(self, test_data_list, answer_predict_dict, compute=False):
        test_data_list.reset()
        self.network.eval()
        pred_json = {}
        for batch in tqdm(test_data_list):
            op_scale = answer_predict_dict[batch['question_uid'][0]][0:2]
            batch['op_scale'] = op_scale
            output_dict = self.network.predict(batch, compute)
            question_id = output_dict["question_uid"]
            pred_answer = output_dict["answer"]
            pred_scale = output_dict["scale"]
            for i in range(len(question_id)):
                if answer_predict_dict[question_id[i]][0] == 2:
                    best_scale = SCALE[answer_predict_dict[question_id[i]][1]]
                    if len(answer_predict_dict[question_id[i]]) == 2:
                        # answer_predict_dict[question_id[i]].append([pred_answer[i], pred_scale[i]])
                        answer_predict_dict[question_id[i]].append([pred_answer[i], best_scale])
                    else:
                        # answer_predict_dict[question_id[i]][2] = [pred_answer[i], pred_scale[i]]
                        answer_predict_dict[question_id[i]][2] = [pred_answer[i], best_scale]

            # 去掉op 和 scale
            for i in range(len(question_id)):
                pred_json[question_id[i]] = answer_predict_dict[question_id[i]][2]

        return pred_json

    @torch.no_grad()
    def predict_arithmetic(self, test_data_list, answer_predict_dict, compute=False):
        test_data_list.reset()
        self.network.eval()
        for batch in tqdm(test_data_list):
            op_scale = answer_predict_dict[batch['question_uid'][0]][0:2]
            batch['op_scale'] = op_scale
            output_dict = self.network.predict(batch, compute)
            question_id = output_dict["question_uid"]
            pred_answer = output_dict["answer"]
            pred_scale = output_dict["scale"]
            for i in range(len(question_id)):
                if answer_predict_dict[question_id[i]][0] == 3:
                    best_scale = SCALE[answer_predict_dict[question_id[i]][1]]
                    if len(answer_predict_dict[question_id[i]]) == 2:
                        # answer_predict_dict[question_id[i]].append([pred_answer[i], pred_scale[i]])
                        answer_predict_dict[question_id[i]].append([pred_answer[i], best_scale])
                    else:
                        # answer_predict_dict[question_id[i]][2] = [pred_answer[i], pred_scale[i]]
                        answer_predict_dict[question_id[i]][2] = [pred_answer[i], best_scale]

        return answer_predict_dict


    def reset(self):
        if self.args.gpu_num > 1:
            self.mnetwork.module.reset()
        else:
            self.mnetwork.reset()

    def get_df(self):
        if self.args.gpu_num > 1:
            self.mnetwork.module.get_df()
        else:
            self.mnetwork.get_df()

    def get_metrics(self, logger=None):
        if self.args.gpu_num > 1:
            return self.mnetwork.module.get_metrics(logger, True)
        else:
            return self.mnetwork.get_metrics(logger, True)


class RegHNTFineTuningModel():
    def __init__(self, args, network, state_dict=None, num_train_steps=1):
        self.args = args
        self.train_loss = AverageMeter()
        self.dev_loss = AverageMeter()
        self.step = 0
        self.updates = 0
        self.network = network
        if state_dict is not None:
            print("Load Model!")
            self.network.load_state_dict(state_dict["state"])
        self.mnetwork = nn.DataParallel(self.network) if args.gpu_num > 1 else self.network
        self.fgm = FGM(self.mnetwork)

        self.total_param = sum([p.nelement() for p in self.network.parameters() if p.requires_grad])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in self.network.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.bert_learning_rate},
            {'params': [p for n, p in self.network.named_parameters() if not n.startswith("encoder.")],
             "weight_decay": args.weight_decay, "lr": args.learning_rate}
        ]
        self.optimizer = Adam(optimizer_parameters,
                              lr=args.learning_rate,
                              warmup=args.warmup,
                              t_total=num_train_steps,
                              max_grad_norm=args.grad_clipping,
                              schedule=args.warmup_schedule)
        if self.args.gpu_num > 0:
            self.network.cuda()

    def avg_reset(self):
        self.train_loss.reset()  # 置零
        self.dev_loss.reset()  # 置零

    def update(self, tasks):
        self.network.train()
        output_dict = self.mnetwork(tasks)  # input:字典,  原先**task
        loss = output_dict["loss"]
        loss = torch.mean(loss)
        self.train_loss.update(loss.item(), 1)
        if self.args.gradient_accumulation_steps > 1:
            loss /= self.args.gradient_accumulation_steps
        loss.backward()  # 梯度下降：反向传播，得到正常的grad

        # 对抗训练
        # self.fgm.attack()  # 在embedding上添加对抗扰动
        # loss_adv = self.mnetwork(tasks)["loss"]
        # if self.args.gradient_accumulation_steps > 1:
        #     loss_adv /= self.args.gradient_accumulation_steps
        # loss_adv.backward()  # 梯度上升：反向传播，并在正常的grad基础上，累加对抗训练的梯度
        # self.fgm.restore()  # 恢复embedding参数

        if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.updates += 1
        self.step += 1

    @torch.no_grad()
    def evaluate(self, dev_data_list):
        dev_data_list.reset()
        self.network.eval()
        for batch in tqdm(dev_data_list):
            output_dict = self.network(batch)  # 原先**batch, mode="eval"
            loss = output_dict["loss"]
            self.dev_loss.update(loss.item(), 1)
        self.network.train()

    @torch.no_grad()
    def predict(self, test_data_list):  # trainer.py  默认compute=T
        test_data_list.reset()
        self.network.eval()
        # pred_json = {}
        for batch in tqdm(test_data_list):
            output_dict = self.network.predict(batch)  # 原先**batch , mode="eval"
            # loss = output_dict["loss"]
            # self.dev_loss.update(loss.item(), 1)

    def reset(self):
        if self.args.gpu_num > 1:
            self.mnetwork.module.reset()
        else:
            self.mnetwork.reset()

    def get_df(self):
        if self.args.gpu_num > 1:
            self.mnetwork.module.get_df()
        else:
            self.mnetwork.get_df()

    def get_metrics(self, logger=None):
        if self.args.gpu_num > 1:
            return self.mnetwork.module.get_metrics(logger, True)
        else:
            return self.mnetwork.get_metrics(logger, True)

    def save(self, prefix, epoch):
        network_state = dict([(k, v.cpu()) for k, v in self.network.state_dict().items()])
        other_params = {
            'optimizer': self.optimizer.state_dict(),
            'config': self.args,
            'epoch': epoch
        }
        state_path = prefix + ".pt"
        other_path = prefix + ".ot"
        torch.save(other_params, other_path)
        torch.save(network_state, state_path)
        print('model saved to {}'.format(prefix))
