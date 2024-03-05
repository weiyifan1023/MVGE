import os
import pickle
import torch
import random
import numpy as np
import dgl


class TaTQABatchGen(object):
    def __init__(self, args, data_mode, encoder='roberta', flag=0):
        # dpath = f"reghnt_{encoder}_cached_{data_mode}.pkl"  # 由pre_pare_dataset.py调用tat qa_dataset.py的TaTQAReader类 生成的pkl
        encoder = "roberta_large"
        dpath = f"coarse_wyf_reghnt_{encoder}_cached_{data_mode}.pkl"  # 服务器  新的多视图数据

        self.is_train = data_mode == "train"
        # self.is_train = flag
        self.args = args
        # with open(os.path.join("./cache", dpath), 'rb') as f:
        with open(os.path.join(args.data_dir, dpath), 'rb') as f:  # 改  服务器版本
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        all_data = []
        for item in data:
            input_ids = torch.from_numpy(item["input_ids"])
            attention_mask = torch.from_numpy(item["attention_mask"])
            token_type_ids = torch.from_numpy(item["token_type_ids"])

            word_tokens_tags = torch.from_numpy(item['word_tokens_tags'])
            word_tokens_lens = item['word_tokens_lens']

            word_word_mask = item['word_word_mask']
            number_word_mask = item['number_word_mask']
            word_subword_mask = torch.tensor(item['word_subword_mask'])
            number_subword_mask = torch.tensor(item['number_subword_mask'])

            question_word_tokens = item['question_word_tokens']
            table_cell_tokens = item['table_cell_tokens']
            paragraph_word_tokens = item['paragraph_word_tokens']
            item_length = item['item_length']

            table_word_tokens = item['table_word_tokens']
            table_cell_tokens_len = item['table_cell_tokens_len']  # 原先table_word_tokens_number

            table_pd = item['table_pd']
            table_cell_type_ids = torch.from_numpy(item['table_cell_type'])
            table_cell_is_head_scale = torch.from_numpy(item['table_cell_is_head_scale'])
            table_subword_index = torch.from_numpy(item['table_subword_index'])
            paragraphs_subword_index = torch.from_numpy(item['paragraphs_subword_index'])

            question_number = item['question_number']
            table_number = item['table_number']
            paragraph_number = item['paragraph_number']

            question_number_index = torch.from_numpy(np.array(item['question_number_index']))
            table_number_index = torch.from_numpy(item['table_number_index'])
            paragraph_number_index = torch.from_numpy(item['paragraph_number_index'])

            table_number_scale = torch.tensor(item['table_number_scale'])
            paragraph_number_scale = torch.tensor(item['paragraph_number_scale'])

            answer_dict = item['answer_dict']
            question_uid = item['question_uid']

            operator_class = torch.tensor(item['operator_class'])
            scale_class = item['scale_class']

            question = item['question']
            answer_tp = item['answer_tp']
            outputs = item['outputs']

            paragraph_sep_tag = item['paragraph_sep_tag']
            p_sep_fuzhu_tag = item['p_sep_fuzhu_tag']

            graph = item['graph']
            multi_graph = item['multi_graph']  # list[graph1,2,3]

            all_data.append((input_ids, attention_mask, token_type_ids,
                             word_tokens_tags, word_tokens_lens,
                             word_word_mask, number_word_mask,
                             word_subword_mask, number_subword_mask,
                             question_word_tokens, table_cell_tokens, paragraph_word_tokens, item_length,
                             table_word_tokens, table_cell_tokens_len, table_pd, table_cell_type_ids, table_cell_is_head_scale,
                             table_subword_index, paragraphs_subword_index,
                             question_number, table_number, paragraph_number,
                             question_number_index, table_number_index, paragraph_number_index,
                             table_number_scale, paragraph_number_scale,
                             question_uid, question, answer_dict, outputs, answer_tp, operator_class, scale_class,
                             graph, multi_graph, paragraph_sep_tag, p_sep_fuzhu_tag))  # 39
        # for end
        print("Load data size {}.".format(len(all_data)))
        self.data = TaTQABatchGen.make_batches(all_data, args.batch_size if self.is_train else args.eval_batch_size, self.is_train)
        self.offset = 0  # data[batch=[instance1,……,instance_bs],batch_i]

    @staticmethod
    def make_batches(data, batch_size=32, is_train=True):
        if is_train:
            random.shuffle(data)
        if is_train:
            return [
                data[i: i + batch_size] if i + batch_size < len(data) else data[i:] + data[:i + batch_size - len(data)]
                for i in range(0, len(data), batch_size)]
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def reset(self):
        if self.is_train:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            for i in range(len(self.data)):
                random.shuffle(self.data[i])
        self.offset = 0
    # Batch == Iteration
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        while self.offset < len(self):
            batch = self.data[self.offset]  # data[list[list (len=4)]]  data[batch=[instance1,……,instance_bs],batch_i]
            self.offset += 1

            input_ids_batch, attention_mask_batch, token_type_ids_batch, \
            word_tokens_tags_batch, word_tokens_lens_batch, \
            word_word_mask_batch, number_word_mask_batch, \
            word_subword_mask_batch, number_subword_mask_batch, \
            question_word_tokens_batch, table_cell_tokens_batch, paragraph_word_tokens_batch, item_length_batch, \
            table_word_tokens_batch, table_cell_tokens_len_batch, table_pd_batch, table_cell_type_ids_batch, table_cell_is_head_scale_batch, \
            table_subword_index_batch, paragraphs_subword_index_batch, \
            question_number_batch, table_number_batch, paragraph_number_batch, \
            question_number_index_batch, table_number_index_batch, paragraph_number_index_batch, \
            table_number_scale_batch, paragraph_number_scale_batch, \
            question_uid_batch, question_batch, answer_dict_batch, outputs_batch, answer_tp_batch, operator_class_batch, scale_class_batch, \
            graph_batch, multi_graph_batch, paragraph_sep_tag_batch, p_sep_fuzhu_tag_batch = zip(*batch)

            #  multi_graph_batch = (t1[38],t2[38],t3[38],t4[38]) 共bs条数据
            bs = len(batch)  # batch_size

            input_ids = torch.LongTensor(bs, 512)
            attention_mask = torch.LongTensor(bs, 512)
            token_type_ids = torch.LongTensor(bs, 512).fill_(0)
            operator_class = torch.LongTensor(bs)
            scale_class = torch.LongTensor(bs)

            table = []
            table_cell_type_ids = []
            table_cell_is_head_scale = []
            question_number = []
            question_number_index = []
            table_number = []
            table_number_index = []
            paragraph_number = []
            paragraph_number_index = []
            answer_dict = []
            question_uid = []
            table_cell_tokens_len = []
            gold_answers = []
            table_number_scale = []
            paragraph_number_scale = []
            questions = []

            word_subword_mask = torch.LongTensor(bs, 512)
            number_subword_mask = torch.LongTensor(bs, 512)
            # max_node_num最大节点数,没加row/column node
            max_batch_node_num = max(list(map(lambda x: x['q'] + x['t'] + x['p'] + x['r'] + x['c'], item_length_batch)))
            max_p_sep_len = max(list(map(lambda x: len(x), p_sep_fuzhu_tag_batch)))

            # new add 初始化 size = bs X batch max node number   batch里最大的节点node个数
            b_mask = torch.zeros([bs, max_batch_node_num], dtype=bool).cuda()  # 原max_batch_len
            b_question_mask = torch.zeros_like(b_mask)
            b_table_mask = torch.zeros_like(b_mask)
            b_paragraph_mask = torch.zeros_like(b_mask)
            b_row_mask = torch.zeros_like(b_mask)
            b_col_mask = torch.zeros_like(b_mask)
            b_number_subword_mask = torch.zeros_like(b_mask)
            b_p_sep_mask = torch.zeros_like(b_mask)
            # mask ↑
            word_tokens_tags = torch.zeros_like(b_mask, dtype=int)
            sep_tag = torch.zeros_like(b_mask).long()
            start_pos = torch.zeros_like(b_mask, dtype=int)
            end_pos = torch.zeros_like(b_mask, dtype=int)
            # size == b_mask.size
            is_t_tag = torch.zeros([bs], dtype=bool).cuda()
            is_p_tag = torch.zeros([bs], dtype=bool).cuda()
            outputs = []
            b_tokens = []
            # new add
            for i in range(bs):
                question_word_tokens = question_word_tokens_batch[i]  # one instance ; bs * instance = batch
                table_cell_tokens = table_cell_tokens_batch[i]
                paragraph_word_tokens = paragraph_word_tokens_batch[i]

                b_tokens.append(question_word_tokens + table_cell_tokens + paragraph_word_tokens)  # tokens=Q +T+ P tokens
                outputs.append(outputs_batch[i])

                input_ids[i] = input_ids_batch[i]
                attention_mask[i] = attention_mask_batch[i]
                token_type_ids[i] = token_type_ids_batch[i]

                operator_class[i] = operator_class_batch[i]
                scale_class[i] = scale_class_batch[i]

                table.append(table_pd_batch[i])
                table_cell_type_ids.append(table_cell_type_ids_batch[i])
                table_cell_is_head_scale.append(table_cell_is_head_scale_batch[i])
                question_number.append(question_number_batch[i])
                question_number_index.append(question_number_index_batch[i])
                table_number.append(table_number_batch[i])
                table_number_index.append(table_number_index_batch[i])
                paragraph_number.append(paragraph_number_batch[i])
                paragraph_number_index.append(paragraph_number_index_batch[i])

                word_subword_mask[i] = word_subword_mask_batch[i]
                number_subword_mask[i] = number_subword_mask_batch[i]

                table_number_scale.append(table_number_scale_batch[i])
                paragraph_number_scale.append(paragraph_number_scale_batch[i])

                questions.append(question_batch[i])
                answer_dict.append(answer_dict_batch[i])
                question_uid.append(question_uid_batch[i])
                gold_answers.append(answer_dict_batch[i]['answer'])  # {'answer_type': 'multi-span', 'answer': ['73,260', '57,768'], 'scale': 'thousand', 'answer_from': 'table'}

                item_l = item_length_batch[i]  # item_length = {'q': 7, 't': 29, 'p': 19}
                leng = item_l['q'] + item_l['t'] + item_l['p'] + item_l['r'] + item_l['c']   # 之后multi-view 需要加入row/column token

                # b_mask部分是 max q、t、p Node Num level
                b_mask[i][:leng] = 1  # real node leng数  置为1;  bs X batch max Node Number
                b_question_mask[i][:item_l['q']] = 1  # Q real node(sub word token)
                b_table_mask[i][item_l['q']:item_l['q'] + item_l['t']] = 1  # T real ...
                b_paragraph_mask[i][item_l['q'] + item_l['t']: item_l['q'] + item_l['t'] + item_l['p']] = 1  # P real ...
                b_row_mask[i][item_l['q'] + item_l['t'] + item_l['p']:item_l['q'] + item_l['t'] + item_l['p'] + item_l['r']] = 1
                b_col_mask[i][item_l['q'] + item_l['t'] + item_l['p'] + item_l['r']:item_l['q'] + item_l['t'] + item_l['p'] + item_l['r'] + item_l['c']] = 1
                # if sub word == <sep>设为1 else 0,长度是q、t、p总长
                b_p_sep_mask[i][:len(paragraph_sep_tag_batch[i])] = torch.tensor(paragraph_sep_tag_batch[i])

                word_tokens_tags[i, :len(word_tokens_tags_batch[i])] = word_tokens_tags_batch[i]  # 总长512，多余为0无影响，不是答案来源
                sep_tag[i][:len(p_sep_fuzhu_tag_batch[i])] = torch.tensor(p_sep_fuzhu_tag_batch[i])  # b_sep_tag

                question_number_index_i = question_number_index_batch[i]  # len = q_number个数,index是在Q_token中的下标
                table_number_index_i = table_number_index_batch[i]
                paragraph_number_index_i = paragraph_number_index_batch[i]
                for idx in question_number_index_i:
                    b_number_subword_mask[i][idx.item()] = 1  # 是数字的index置为1
                for idx in table_number_index_i:
                    b_number_subword_mask[i][idx.item() + item_l['q']] = 1  # len=512
                for idx in paragraph_number_index_i:
                    b_number_subword_mask[i][idx.item() + item_l['q'] + item_l['t']] = 1
                mask = {"b": b_mask, "question": b_question_mask, "table": b_table_mask, "paragraph": b_paragraph_mask,
                        "row": b_row_mask, "col": b_col_mask, "number": b_number_subword_mask, "sep": b_p_sep_mask}

                is_t_tag[i] = torch.tensor(answer_tp_batch[i][0])
                is_p_tag[i] = torch.tensor(answer_tp_batch[i][1])
                if operator_class[i].item() == 0:
                    find_pos = torch.where(word_tokens_tags_batch[i] == 1)[0].tolist()
                    if len(find_pos) > 0:
                        if len(find_pos) == 1:
                            start_pos[i][find_pos[0]] = 1
                            end_pos[i][find_pos[0]] = 1
                        else:
                            start_pos[i][find_pos[0]] = 1
                            end_pos[i][find_pos[-1]] = 1

            # mask指哪些是type=2 or 3 数字和词type
            b_word_word_mask = torch.cat([torch.tensor(word_word_mask) for word_word_mask in word_word_mask_batch]).bool()  # list + list
            b_number_word_mask = torch.cat([torch.tensor(number_word_mask) for number_word_mask in number_word_mask_batch]).bool()
            b_word_tokens_lens = torch.cat([torch.tensor(word_tokens_lens) for word_tokens_lens in word_tokens_lens_batch])
            # 3b是原始输入，以下是mask过滤后的
            word_word_lens = b_word_tokens_lens.masked_select(b_word_word_mask)  # 序列，只获取word 的len
            number_word_lens = b_word_tokens_lens.masked_select(b_number_word_mask)  # 只获取number的 word level len

            max_word_word_len = max(word_word_lens)  # batch里 word：word可以分解的最大个数
            max_number_word_len = max(number_word_lens)  # batch里 number：word可以分解的最大个数

            # word_word num X max word contains sub_word tokens num
            word_word_num_X_max_word_len_mask = torch.zeros([word_word_lens.size(0), max_word_word_len], dtype=bool)  # 原先word_subword_mask
            number_word_num_X_max_word_len_mask = torch.zeros([number_word_lens.size(0), max_number_word_len], dtype=bool)
            for i in range(len(word_word_num_X_max_word_len_mask)):
                word_word_num_X_max_word_len_mask[i][:word_word_lens[i]] = 1
            for i in range(len(number_word_num_X_max_word_len_mask)):
                number_word_num_X_max_word_len_mask[i][:number_word_lens[i]] = 1

            # t_w_t_n_mask
            table_cell_tokens_len_lengths = 0  # cell总个数
            max_table_cell_tokens_len = 0  # batch里 cell包含的word token数最大的值
            word_tokens_number = []  # 每个cell包含多少个word token
            for table_cell_tokens_len in table_cell_tokens_len_batch:
                table_cell_tokens_len_lengths += len(table_cell_tokens_len)  # total length for batch,batch一共有几个cell
                max_table_cell_tokens_len = max(max_table_cell_tokens_len, max(table_cell_tokens_len))  #
                word_tokens_number += table_cell_tokens_len
            # table word token number?
            t_w_t_n_mask = torch.zeros([table_cell_tokens_len_lengths, max_table_cell_tokens_len]).bool().cuda()
            for i in range(len(t_w_t_n_mask)):
                t_w_t_n_mask[i][:word_tokens_number[i]] = 1  # cell num X max cell contains word tokens num

            # max node num len部分
            max_question_len = max([item_l['q'] for item_l in item_length_batch])  # max Q word token num
            max_table_len = max([item_l['t'] for item_l in item_length_batch])  # max T cell token num
            max_paragraph_len = max([item_l['p'] for item_l in item_length_batch])  # max P word token num
            max_row_len = max([item_l['r'] for item_l in item_length_batch])  # max row node
            max_col_len = max([item_l['c'] for item_l in item_length_batch])  # max col node
            # max T word token num
            max_table_word_len = max([len(word_tokens_lens_batch[i])-item_l['p']-item_l['q'] for i, item_l in enumerate(item_length_batch)])
            max_len = {"question": max_question_len, "table": max_table_len, "table_word": max_table_word_len,
                       "paragraph": max_paragraph_len, "row": max_row_len, "col": max_col_len}

            # q、t、p real len部分,即 node num个数
            question_lens = torch.tensor([item_l['q'] for item_l in item_length_batch]).cuda()
            table_lens = torch.tensor([item_l['t'] for item_l in item_length_batch]).cuda()  # T cell token num tensor
            paragraph_lens = torch.tensor([item_l['p'] for item_l in item_length_batch]).cuda()
            row_lens = torch.tensor([item_l['r'] for item_l in item_length_batch]).cuda()
            col_lens = torch.tensor([item_l['c'] for item_l in item_length_batch]).cuda()
            table_word_lens = torch.tensor([len(word_tokens_lens_batch[i]) - item_l['p'] - item_l['q']
                                            for i, item_l in enumerate(item_length_batch)]).cuda()
            lens = {"question": question_lens, "table": table_lens, "table_word": table_word_lens,
                    "paragraph": paragraph_lens, "row": row_lens, "col": col_lens}

            # pl_mask 部分  word token level mask
            max_pl_len = max([len(word_tokens_lens) for word_tokens_lens in word_tokens_lens_batch])  # batch最大的total word token 个数
            b_pl_mask = torch.zeros([bs, max_pl_len]).bool().cuda()  # bs * max batch total word token number
            q_pl_mask = torch.zeros_like(b_pl_mask)
            t_pl_mask = torch.zeros_like(b_pl_mask)
            p_pl_mask = torch.zeros_like(b_pl_mask)
            question_pl_mask = torch.zeros([bs, max_question_len]).bool().cuda()  # bs * max Q word token num
            table_pl_mask = torch.zeros([bs, max_table_word_len]).bool().cuda()  # bs * max T word token num
            paragraph_pl_mask = torch.zeros([bs, max_paragraph_len]).bool().cuda()  # bs * max P word token num
            cell_pl_mask = torch.zeros([bs, max_table_len]).bool().cuda()  # bs * max T cell token num
            row_col_pl_mask = torch.zeros([bs, max_row_len, max_col_len]).bool().cuda()  # bs * max row * max col
            for i in range(bs):
                b_pl_mask[i][:question_lens[i] + table_word_lens[i] + paragraph_lens[i]] = 1
                question_pl_mask[i][:question_lens[i]] = 1  # bs * max Q word token num
                table_pl_mask[i][:table_word_lens[i]] = 1
                paragraph_pl_mask[i][:paragraph_lens[i]] = 1
                cell_pl_mask[i][:table_lens[i]] = 1  # wyf add ; bs * max  cell token num
                # 特别处理，存在row，col 矩阵内无内容的cell
                row_col_pl_mask[i][:row_lens[i], :col_lens[i]] = 1  # 多维切片
                for r in range(len(table_cell_type_ids[i])):
                    for c in range(len(table_cell_type_ids[i][0])):
                        # print(np.array(table_cell_type_ids).shape())
                        if table_cell_type_ids[i][r][c] == 0:  # 0是cell为空  -1是cell无主要内
                            # print(row_col_pl_mask.size())
                            # print("batch id:{}, row id:{}, col id:{} ".format(i, len(table_cell_type_ids[i]), len(table_cell_type_ids[i][0])))
                            row_col_pl_mask[i][r][c] = 0

                q_pl_mask[i][:question_lens[i]] = 1  # bs * max batch total Word Token number
                t_pl_mask[i][question_lens[i]:question_lens[i] + table_word_lens[i]] = 1
                p_pl_mask[i][question_lens[i] + table_word_lens[i]:question_lens[i] + table_word_lens[i] + paragraph_lens[i]] = 1
            pl_mask = {"b": b_pl_mask, "question": question_pl_mask, "table": table_pl_mask, "paragraph": paragraph_pl_mask,
                       "cell": cell_pl_mask, "row_col": row_col_pl_mask, "q": q_pl_mask, "t": t_pl_mask, "p": p_pl_mask}

            # Graph部分
            b_graph = dgl.batch([g['dgl'] for g in graph_batch]).to("cuda:0")
            b_relation = torch.cat([torch.tensor(g['relations']) for g in graph_batch]).cuda()
            b_src = torch.cat([torch.tensor(g['src']) for g in graph_batch]).cuda()
            b_dst = torch.cat([torch.tensor(g['dst']) for g in graph_batch]).cuda()

            max_node_num = 0
            for g in multi_graph_batch:  # 获取batch中最大的node数
                node_num = g[0].shape[0]
                max_node_num = max(node_num, max_node_num)
            #  对齐二维邻接矩阵
            b_quantity_graph = [np.pad(g[0], ((0, max_node_num-g[0].shape[0]), (0, max_node_num-g[0].shape[0])), 'constant',constant_values=(0, 0)) for g in multi_graph_batch]  # 子图邻接矩阵 size不一样
            b_spatial_graph = [np.pad(g[1], ((0, max_node_num-g[1].shape[0]), (0, max_node_num-g[1].shape[0])), 'constant',constant_values=(0, 0)) for g in multi_graph_batch]
            b_relation_graph = [np.pad(g[2], ((0, max_node_num-g[2].shape[0]), (0, max_node_num-g[2].shape[0])), 'constant',constant_values=(0, 0)) for g in multi_graph_batch]

            b_quantity_graph = torch.tensor(b_quantity_graph).unsqueeze(1)
            b_spatial_graph = torch.tensor(b_spatial_graph).unsqueeze(1)
            b_relation_graph = torch.tensor(b_relation_graph).unsqueeze(1)
            b_multi_graph = torch.cat((b_quantity_graph, b_spatial_graph, b_relation_graph), 1)

            out_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                         "b_tokens": b_tokens, "word_tokens_tags": word_tokens_tags,
                         "word_word_lens": word_word_lens, "number_word_lens": number_word_lens,
                         "b_word_word_mask": b_word_word_mask, "b_number_word_mask": b_number_word_mask, "t_w_t_n_mask": t_w_t_n_mask,
                         "word_subword_mask": word_subword_mask, "number_subword_mask": number_subword_mask,
                         "word_word_num_X_max_word_len_mask": word_word_num_X_max_word_len_mask,
                         "number_word_num_X_max_word_len_mask": number_word_num_X_max_word_len_mask,
                         "lens": lens, "max_len": max_len, "pl_mask": pl_mask, "mask": mask,
                         "operator_class": operator_class, "scale_class": scale_class,
                         "table": table, "table_cell_type_ids": table_cell_type_ids, "table_cell_is_head_scale": table_cell_is_head_scale,
                         "question_number": question_number, "table_number": table_number, "paragraph_number": paragraph_number,
                         "question_number_index": question_number_index, "table_number_index": table_number_index, "paragraph_number_index": paragraph_number_index,
                         "t_scale": table_number_scale, "p_scale": paragraph_number_scale,
                         "question_uid": question_uid, "questions": questions, "answer_dict": answer_dict, "outputs": outputs,  "gold_answers": gold_answers,
                         "sep_tag": sep_tag, "is_t_tag": is_t_tag, "is_p_tag": is_p_tag, "start_pos": start_pos, "end_pos": end_pos,
                         "b_relation": b_relation, "b_src": b_src, "b_dst": b_dst, "b_graph": b_graph,
                         "b_quantity_graph": b_quantity_graph, "b_spatial_graph": b_spatial_graph, "b_relation_graph": b_relation_graph, "b_multi_graph": b_multi_graph
                         }

            if self.args.cuda:
                for k in out_batch.keys():
                    if isinstance(out_batch[k], torch.Tensor):
                        out_batch[k] = out_batch[k].cuda()

            yield out_batch
