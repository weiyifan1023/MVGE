import torch
import torch.nn as nn
from tatqa_metric import TaTQAEmAndF1
from .tools.util import FFNLayer
from .tools import allennlp as util
from typing import Dict, List, Tuple
import numpy as np
from reg_hnt.data.file_utils import is_scatter_available
import math
from reg_hnt.reghnt.model_utils import rnn_wrapper, lens2mask, PoolingFunction, FFN
from reg_hnt.reghnt.new_model import Arithmetic, SubwordAggregation, InputRNNLayer, RGTAT, OtherModel, MVGTAT
from reg_hnt.reghnt.modeling_tree import Prediction
from reg_hnt.reghnt.Vocabulary import *
from reg_hnt.data.tatqa_dataset import is_year, clean_year

np.set_printoptions(threshold=np.inf)
if is_scatter_available():
    from torch_scatter import scatter
    from torch_scatter import scatter_max


class RegHNTModel(nn.Module):
    def __init__(self,
                 encoder,
                 config,
                 bsz,
                 operator_classes: int,
                 scale_classes: int,
                 operator_criterion: nn.CrossEntropyLoss = None,
                 scale_criterion: nn.CrossEntropyLoss = None,
                 hidden_size: int = None,
                 dropout_prob: float = None,
                 arithmetic_op_index: List = None,
                 op_mode: int = None,
                 ablation_mode: int = None,
                 ):
        super(RegHNTModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.operator_classes = operator_classes
        self.scale_classes = scale_classes
        if hidden_size is None:
            hidden_size = self.config.hidden_size
            self.hidden_size = hidden_size
        if dropout_prob is None:
            dropout_prob = self.config.hidden_dropout_prob

        self.operator_predictor = FFNLayer(hidden_size, hidden_size, operator_classes, dropout_prob)

        self.scale_predictor = FFNLayer(3 * hidden_size, hidden_size, scale_classes, dropout_prob)

        self.dropout_layer = nn.Dropout(p=0.5)

        self.subword_aggregation = SubwordAggregation(hidden_size, subword_aggregation="attentive-pooling")

        self.operator_criterion = operator_criterion
        self.scale_criterion = scale_criterion
        self.config = config
        self.rnn_layer = InputRNNLayer(hidden_size, hidden_size, cell='lstm', schema_aggregation='attentive-pooling')
        # self.hidden_layer = RGTAT(gnn_hidden_size=hidden_size, gnn_num_layers=8, num_heads=8, relation_num=len(Relation_vocabulary['word2id']))
        self.graph_layer = MVGTAT(gnn_hidden_size=hidden_size)
        self.arithmetic_model = Arithmetic(hidden_size)
        self.other_model = OtherModel(hidden_size)

        self.arithmetic_op_index = arithmetic_op_index
        self._metrics = TaTQAEmAndF1()

    """
     :parameter
     input_ids, shape:[bsz, 512] split_tokens' ids, 0 for padded token.
     attention_mask, shape:[bsz, 512] 0 for padded token and 1 for others
     token_type_ids, shape[bsz, 512, 3].
     # row_ids and column_ids are non-zero for table-contents and 0 for others, including headers.
         segment_ids[:, :, 0]: 1 for table and 0 for others
         column_ids[:, :, 1]: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
       tokens, special tokens and padding.
         row_ids[:, :, 2]: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
       special tokens and padding. Tokens of column headers are also 0.
     paragraph_mask, shape[bsz, 512] 1 for paragraph_tokens and 0 for others
     paragraph_index, shape[bsz, 512] 0 for non-paragraph tokens and index starting from 1 for paragraph tokens
     tag_labels: [bsz, 512] 1 for tokens in the answer and 0 for others
     operator_labels: [bsz, 8]
     scale_labels: [bsz, 8]
     number_order_labels: [bsz, 2]
     paragraph_tokens: [bsz, text_len], white-space split tokens
     tables: [bsz,] raw tables in DataFrame.
     paragraph_numbers: [bsz, text_len], corresponding number extracted from tokens, nan for non-number token
     table_numbers: [bsz, table_size], corresponding number extracted from table cells, nan for non-number cell. Shape is the same as flattened table.
     """

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        operator_class = batch['operator_class']
        question_uids = batch['question_uid']

        position_ids = None
        device = input_ids.device

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,  # 可选。就是 token 对应的句子id，值为0或1（0表示对应的token属于第一句，1表示属于第二句）。形状为(bs, sen_length)
            position_ids=position_ids)

        sequence_output = outputs[0]  # last_hidden_state 模型最后一层输出的隐藏状态 shape是(batch_size, sequence_length, hidden_size)
        batch_size = sequence_output.size(0)

        question, table, paragraph = self.subword_aggregation(sequence_output, batch)  # bs * S * hidden_size
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "paragraph": self.dropout_layer(paragraph)
        }

        inputs, graph_nodes = self.rnn_layer(input_dict, batch)  # input:{∑ sen_bs_i q+t+p} * hidden_size; inputs_lengths: list=>q or t or p S_len * 3 * bs

        # inputs = self.hidden_layer(inputs, batch)  # RGAT  graph enhance
        q_mean = torch.mean(question, dim=1, keepdim=True)
        hidden_output = self.graph_layer(q_mean, graph_nodes, batch)  # MVG graph enhance

        # hidden = inputs.new_zeros(batch_size, batch['mask']['b'].size(1), self.hidden_size)
        # hidden_output = hidden.masked_scatter_(batch['mask']['b'].unsqueeze(-1), inputs)

        question_hidden = util.replace_masked_values(hidden_output, batch['mask']['question'].unsqueeze(-1).long(), 0)
        table_hidden = util.replace_masked_values(hidden_output, batch['mask']['table'].unsqueeze(-1).long(), 0)
        paragraph_hidden = util.replace_masked_values(hidden_output, batch['mask']['paragraph'].unsqueeze(-1).long(), 0)
        question_mean = torch.mean(question_hidden, dim=1)
        table_mean = torch.mean(table_hidden, dim=1)
        paragraph_mean = torch.mean(paragraph_hidden, dim=1)

        q_t_p_mean = torch.cat((question_mean, table_mean, paragraph_mean), dim=-1)
        scale_prediction = self.scale_predictor(q_t_p_mean)
        operator_prediction = self.operator_predictor(sequence_output[:, 0])  # cls_output

        outputs = batch['outputs']
        operator_class = batch['operator_class'].tolist()

        #####################################################################################################
        arithmetic_labels = list(map(lambda i: 1 if operator_class[i] == OPERATOR_CLASSES["ARITHMETIC"] else 0, range(len(operator_class))))
        arithmetic_num = arithmetic_labels.count(1)
        other_labels = list([1 for i in range(batch_size)])

        sequence_output_ari, hidden_output_ari = [], []

        for i in range(batch_size):
            if arithmetic_labels[i] == 1:
                sequence_output_ari.append(sequence_output[i])
                hidden_output_ari.append(hidden_output[i])

        if arithmetic_num > 0:
            sequence_output_ari = torch.stack(sequence_output_ari, dim=0)
            hidden_output_ari = torch.stack(hidden_output_ari, dim=0)

        ######################################################################################################
        arithmetic_loss, other_loss = 0, 0
        if arithmetic_num > 0:
            arithmetic_loss, all_node_outputs, target = self.arithmetic_model(sequence_output_ari, hidden_output_ari, batch,
                                                                              arithmetic_labels)

        other_loss, table_tag_prediction, paragraph_tag_prediction = self.other_model(sequence_output, hidden_output, batch, other_labels)

        operator_prediction_loss = self.operator_criterion(operator_prediction, torch.tensor(operator_class).cuda())

        scale_class = batch['scale_class']
        scale_prediction_loss = self.scale_criterion(scale_prediction, scale_class)

        output_dict = {}
        output_dict['loss'] = arithmetic_loss + other_loss + operator_prediction_loss + scale_prediction_loss
        output_dict["question_uid"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []

        LOss = output_dict["loss"].item()
        if np.isnan(LOss):
            import pdb;
            pdb.set_trace()

        with torch.no_grad():

            predicted_operator_class = torch.argmax(operator_prediction, dim=-1).tolist()
            predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()

            for i in range(len(predicted_operator_class)):
                if predicted_operator_class[i] == OPERATOR_CLASSES["ARITHMETIC"]:
                    sequence_output_i = sequence_output[i].unsqueeze(0)
                    hidden_output_i = hidden_output[i].unsqueeze(0)
                    batch_label_i = list(map(lambda x: 1 if x == i else 0, range(batch_size)))
                    answer = self.arithmetic_model.predict(sequence_output_i, hidden_output_i, batch, batch_label_i,
                                                           int(predicted_scale_class[i]), batch['answer_dict'][i])
                else:
                    sequence_output_i = sequence_output[i].unsqueeze(0)
                    hidden_output_i = hidden_output[i].unsqueeze(0)
                    batch_label_i = list(map(lambda x: 1 if x == i else 0, range(batch_size)))
                    answer = self.other_model.predict(hidden_output_i, batch, batch_label_i, predicted_operator_class[i],
                                                      table_tag_prediction[i].unsqueeze(0), paragraph_tag_prediction[i].unsqueeze(0))

                # predict内容,保持公平一致
                # if "which year " in batch['questions'][0] or "Which year " in batch['questions'][0] or "Which FY " in batch['questions'][0] \
                #         or "which FY " in batch['questions'][0]:
                #     if isinstance(answer, list) and len(answer) == 1 and is_year(answer[0]):
                #         answer[0] = clean_year(answer[0])
                #
                # if "which years" in batch['questions'][0] or "Which years" in batch['questions'][0]:
                #     if isinstance(answer, list) and len(answer) > 1:
                #         for i in range(len(answer)):
                #             if is_year(answer[i]):
                #                 answer[i] = clean_year(answer[i])
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                if not answer:
                    answer = -1

                output_dict["answer"].append(answer)
                output_dict["scale"].append(SCALE[int(predicted_scale_class[i])])
                output_dict["question_uid"].append(question_uids[i])

                self._metrics(batch['answer_dict'][i], answer, SCALE[int(predicted_scale_class[i])])

        return output_dict

    def predict(self, batch, compute=True):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        question_uids = batch['question_uid']
        position_ids = None

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)

        question, table, paragraph = self.subword_aggregation(sequence_output, batch)
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "paragraph": self.dropout_layer(paragraph)
        }

        # inputs = self.rnn_layer(input_dict, batch) # 原先
        # inputs = self.hidden_layer(inputs, batch)  # RGAT  graph enhance
        # hidden_output = self.graph_layer(graph_nodes, batch)  # MVG graph enhance

        inputs, graph_nodes = self.rnn_layer(input_dict, batch)
        q_mean = torch.mean(question, dim=1, keepdim=True)
        hidden_output = self.graph_layer(q_mean, graph_nodes, batch)  # MVG graph enhance

        # hidden = inputs.new_zeros(batch_size, batch['mask']['b'].size(1), self.hidden_size)
        # hidden_output = hidden.masked_scatter_(batch['mask']['b'].unsqueeze(-1), inputs)

        question_hidden = util.replace_masked_values(hidden_output, batch['mask']['question'].unsqueeze(-1).long(), 0)
        question_mean = torch.mean(question_hidden, dim=1)
        table_hidden = util.replace_masked_values(hidden_output, batch['mask']['table'].unsqueeze(-1).long(), 0)
        table_mean = torch.mean(table_hidden, dim=1)
        paragraph_hidden = util.replace_masked_values(hidden_output, batch['mask']['paragraph'].unsqueeze(-1).long(), 0)
        paragraph_mean = torch.mean(paragraph_hidden, dim=1)

        q_t_p_mean = torch.cat((question_mean, table_mean, paragraph_mean), dim=-1)
        scale_prediction = self.scale_predictor(q_t_p_mean)
        operator_prediction = self.operator_predictor(sequence_output[:, 0])

        output_dict = {}
        output_dict["question_uid"] = []
        output_dict["answer"] = []
        output_dict["scale"] = []
        output_dict["operator"] = []
        output_dict["gold_answers"] = []
        output_dict['loss'] = []
        ######################################################################################################
        # loss 可用
        ######################################################################################################
        # if compute:
        #     arithmetic_labels = list(map(lambda i: 1 if batch['operator_class'].tolist()[i] == OPERATOR_CLASSES["ARITHMETIC"] else 0,
        #                                  range(len(batch['operator_class'].tolist()))))
        #     arithmetic_num = arithmetic_labels.count(1)
        #     sequence_output_ari, hidden_output_ari = [], []
        #     for i in range(batch_size):
        #         if arithmetic_labels[i] == 1:
        #             sequence_output_ari.append(sequence_output[i])
        #             hidden_output_ari.append(hidden_output[i])
        #     if arithmetic_num > 0:
        #         sequence_output_ari = torch.stack(sequence_output_ari, dim=0)
        #         hidden_output_ari = torch.stack(hidden_output_ari, dim=0)
        #     arithmetic_loss, other_loss = 0, 0
        #     if arithmetic_num > 0:
        #         arithmetic_loss, all_node_outputs, target = self.arithmetic_model(sequence_output_ari, hidden_output_ari, batch,
        #                                                                           arithmetic_labels)
        #     other_loss, table_tag_prediction, paragraph_tag_prediction = self.other_model(sequence_output, hidden_output, batch, [1])
        #     operator_prediction_loss = self.operator_criterion(operator_prediction, torch.tensor(batch['operator_class'].tolist()).cuda())
        #     scale_prediction_loss = self.scale_criterion(scale_prediction, batch['scale_class'])
        #     output_dict['loss'] = arithmetic_loss + other_loss + operator_prediction_loss + scale_prediction_loss
        #     ######################################################################################################

        #   with torch.no_grad():
        predicted_operator_class = torch.argmax(operator_prediction, dim=-1).tolist()
        predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()

        answer = -1

        if predicted_operator_class[0] == OPERATOR_CLASSES["ARITHMETIC"]:
            answer = self.arithmetic_model.predict(sequence_output, hidden_output, batch, [1], int(predicted_scale_class[0]), 0)
        else:
            _, table_tag_prediction, paragraph_tag_prediction = self.other_model.forward2(sequence_output, hidden_output, batch, [1])
            answer = self.other_model.predict(hidden_output, batch, [1], predicted_operator_class[0], table_tag_prediction,
                                              paragraph_tag_prediction)

        if "which year " in batch['questions'][0] or "Which year " in batch['questions'][0] or "Which FY " in batch['questions'][
            0] or "which FY " in batch['questions'][0]:
            if isinstance(answer, list) and len(answer) == 1 and is_year(answer[0]):
                answer[0] = clean_year(answer[0])

        if "which years" in batch['questions'][0] or "Which years" in batch['questions'][0]:
            if isinstance(answer, list) and len(answer) > 1:
                for i in range(len(answer)):
                    if is_year(answer[i]):
                        answer[i] = clean_year(answer[i])

        # predict
        output_dict["answer"].append(answer)
        output_dict["scale"].append(SCALE[int(predicted_scale_class[0])])
        output_dict["operator"].append(predicted_operator_class[0])
        # gold
        output_dict["question_uid"].append(question_uids[0])

        # predict:T ; predict2: F
        if compute:
            output_dict["gold_answers"].append(batch['answer_dict'][0])  # test没有这部分
            current_op = OPERATOR_CLASSES_R[predicted_operator_class[0]]
            gold_op = OPERATOR_CLASSES_R[batch['operator_class'].tolist()[0]]

            self._metrics({**batch['answer_dict'][0], "uid": question_uids[0]}, answer,
                          SCALE[int(predicted_scale_class[0])], None, None, pred_op=current_op, gold_op=gold_op)

        return output_dict

    # def predict2(self, **batch):  # 不计算F1
    #     input_ids = batch['input_ids']
    #     attention_mask = batch['attention_mask']
    #     token_type_ids = batch['token_type_ids']
    #     question_uids = batch['question_uid']
    #     position_ids = None
    #
    #     outputs = self.encoder(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids)
    #
    #     sequence_output = outputs[0]
    #     batch_size = sequence_output.size(0)
    #
    #     question, table, paragraph = self.subword_aggregation(sequence_output, batch)
    #     input_dict = {
    #         "question": self.dropout_layer(question),
    #         "table": self.dropout_layer(table),
    #         "paragraph": self.dropout_layer(paragraph)
    #     }
    #
    #     # inputs = self.rnn_layer(input_dict, batch)
    #     # inputs = self.hidden_layer(inputs, batch)
    #
    #     inputs, graph_nodes = self.rnn_layer(input_dict, batch)
    #     q_mean = torch.mean(question, dim=1, keepdim=True)
    #     hidden_output = self.graph_layer(q_mean, graph_nodes, batch)  # MVG graph enhance
    #
    #     # hidden = inputs.new_zeros(batch_size, batch['mask']['b'].size(1), self.hidden_size)
    #     # hidden_output = hidden.masked_scatter_(batch['mask']['b'].unsqueeze(-1), inputs)
    #
    #     question_hidden = util.replace_masked_values(hidden_output, batch['mask']['question'].unsqueeze(-1).long(), 0)
    #     question_mean = torch.mean(question_hidden, dim=1)
    #     table_hidden = util.replace_masked_values(hidden_output, batch['mask']['table'].unsqueeze(-1).long(), 0)
    #     table_mean = torch.mean(table_hidden, dim=1)
    #     paragraph_hidden = util.replace_masked_values(hidden_output, batch['mask']['paragraph'].unsqueeze(-1).long(), 0)
    #     paragraph_mean = torch.mean(paragraph_hidden, dim=1)
    #
    #     q_t_p_mean = torch.cat((question_mean, table_mean, paragraph_mean), dim=-1)
    #     scale_prediction = self.scale_predictor(q_t_p_mean)
    #     operator_prediction = self.operator_predictor(sequence_output[:, 0])
    #
    #
    #     predicted_operator_class = torch.argmax(operator_prediction, dim=-1).tolist()
    #     predicted_scale_class = torch.argmax(scale_prediction, dim=-1).detach().cpu().numpy()
    #
    #     output_dict = {}
    #
    #     output_dict["question_uid"] = []
    #     output_dict["answer"] = []
    #     output_dict["scale"] = []
    #     output_dict["operator"] = []
    #     answer = -1
    #     if predicted_operator_class[0] == OPERATOR_CLASSES["ARITHMETIC"]:
    #         answer = self.arithmetic_model.predict(sequence_output, hidden_output, batch, [1], int(predicted_scale_class[0]), 0)
    #     else:
    #         _, table_tag_prediction, paragraph_tag_prediction = self.other_model.forward2(sequence_output, hidden_output, batch, [1])
    #         answer = self.other_model.predict(hidden_output, batch, [1], predicted_operator_class[0], table_tag_prediction, paragraph_tag_prediction)
    #
    #
    #     if "which year " in batch['questions'][0] or "Which year " in batch['questions'][0] or "Which FY " in batch['questions'][0] or "which FY " in batch['questions'][0]:
    #         if isinstance(answer, list) and len(answer) == 1 and is_year(answer[0]):
    #             answer[0] = clean_year(answer[0])
    #
    #     if "which years" in batch['questions'][0] or "Which years" in batch['questions'][0]:
    #         if isinstance(answer, list) and len(answer) > 1:
    #             for i in range(len(answer)):
    #                 if is_year(answer[i]):
    #                     answer[i] = clean_year(answer[i])
    #
    #
    #     output_dict["answer"].append(answer)
    #     output_dict["scale"].append(SCALE[int(predicted_scale_class[0])])
    #     output_dict["question_uid"].append(question_uids[0])
    #     output_dict["operator"].append(predicted_operator_class[0])
    #     return output_dict

    def reset(self):
        self._metrics.reset()

    def set_metrics_mdoe(self, mode):
        self._metrics = TaTQAEmAndF1(mode=mode)

    def get_metrics(self, logger=None, reset: bool = False) -> Dict[str, float]:
        detail_em, detail_f1 = self._metrics.get_detail_metric()

        raw_detail = self._metrics.get_raw_pivot_table()
        exact_match, f1_score, scale_score, op_score = self._metrics.get_overall_metric(reset)

        arithmetic_number, count_number, multi_number, span_number = list(raw_detail.loc['arithmetic']), list(raw_detail.loc['count']), list(raw_detail.loc['multi-span']), list(raw_detail.loc['span'])
        arithmetic_em_d, count_em_d, multi_em_d, span_em_d  = list(detail_em.loc['arithmetic']), list(detail_em.loc['count']), list(detail_em.loc['multi-span']), list(detail_em.loc['span'])
        arithmetic_em, count_em, multi_em, span_em = \
            np.sum(np.multiply(arithmetic_number, arithmetic_em_d)) / np.sum(arithmetic_number),\
            np.sum(np.multiply(count_number, count_em_d)) / np.sum(count_number), \
            np.sum(np.multiply(multi_number, multi_em_d)) / np.sum(multi_number), \
            np.sum(np.multiply(span_number, span_em_d)) / np.sum(span_number)


        print(f"raw matrix:{raw_detail}\r\n")
        print(f"detail em:{detail_em}\r\n")
        print(f"detail f1:{detail_f1}\r\n")
        print(f"global em:{exact_match}\r\n")
        print(f"global f1:{f1_score}\r\n")
        print(f"global scale:{scale_score}\r\n")
        print(f"global op:{op_score}\r\n")
        print(f"arithmetic em:{arithmetic_em}\r\n")
        print(f"count em:{count_em}\r\n")
        print(f"multi em:{multi_em}\r\n")
        print(f"span em:{span_em}\r\n")
        if logger is not None:
            logger.info(f"raw matrix:{raw_detail}\r\n")
            logger.info(f"detail em:{detail_em}\r\n")
            logger.info(f"detail f1:{detail_f1}\r\n")
            logger.info(f"global em:{exact_match}\r\n")
            logger.info(f"global f1:{f1_score}\r\n")
            logger.info(f"global scale:{scale_score}\r\n")
            logger.info(f"global op:{op_score}\r\n")
            logger.info(f"arithmetic em:{arithmetic_em}\r\n")
            logger.info(f"count em:{count_em}\r\n")
            logger.info(f"multi em:{multi_em}\r\n")
            logger.info(f"span em:{span_em}\r\n")
        return {'em': exact_match, 'f1': f1_score, "scale": scale_score, "op": op_score, "arithmetic_em":arithmetic_em, "count_em":count_em,
                "multi_em":multi_em, "span_em":span_em}

    def get_df(self):
        raws = self._metrics.get_raw()
        detail_em, detail_f1 = self._metrics.get_detail_metric()
        raw_detail = self._metrics.get_raw_pivot_table()
        return detail_em, detail_f1, raws, raw_detail



