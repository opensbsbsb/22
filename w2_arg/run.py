#  王潞翔
#  时间：2023/2/9 9:11
import argparse
from config import Config
import torch
import data_loader
import utils
from torch.utils.data import DataLoader
from tqdm import tqdm


import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
import numpy as np
import prettytable as pt
import json
'''
原版：标签和触发词double
ECA通道注意力加在多粒度膨胀分组二维卷积神经网络的卷积后，激活函数前
（可 选：不加边缘算子：oneConv、 边缘算子不带Linear：twoConv_nolinear、 边缘算子Linear：twoConv_linear）
'''
DEVICE = 0
def Dic(config):
    event_role_dic = {}
    with open('./data/{}/event_role.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        event_role = json.load(f)
    for tri in event_role:
        a = config.vocab.trilabel_to_id(tri)
        b = []
        for role in event_role[tri]:
            b.append(config.vocab.arglabel_to_id(role))
        event_role_dic[a] = b
    return event_role_dic

class Trainer(object):
    def __init__(self, model, config, updates_total):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader, event_role_dic, config, model, logger):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        total_arg_r = 0
        total_arg_p = 0
        total_arg_c = 0
        total_arg_r_span = 0
        total_arg_p_span = 0
        total_arg_c_span = 0

        for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Training"):
            entity_text = data_batch[-4]
            tri_index = data_batch[-2]
            data_batch = [data_batch[0].cuda(), data_batch[1].cuda(), data_batch[2].cuda(), data_batch[3].cuda(),
                          data_batch[4].cuda(), data_batch[5].cuda(), data_batch[-3].cuda(), data_batch[-1]]
            '''
            bert_inputs:[B, L']
            grid_labels, grid_mask2d, dist_inputs:[B, L, L]
            pieces2word:[B, L, L']
            '''
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, tri_inputs, tri_types = data_batch

            '''
            需要传入模型进行训练的有bert_inputs，grid_mask2d，dist_inputs，pieces2word，sent_length
            '''
            # outputs [B, L, L, num_class] num_class指的是有多少类实体
            outputs = model.forward(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, tri_inputs, tri_index, tri_types, event_role_dic, config)

            '''
            # 处理outputs的约束
            '''
            # for ii, jj in enumerate(tri_types):
            #     roles = event_role_dic[jj]
            #     roles.append(0)
            #     roles.append(1)
            #     array = np.array(range(0, config.vocab.arg_label2id.__len__()))
            #     no_roles = set(array) - set(roles)
            #     for pp in no_roles:
            #         outputs[ii, :, :, pp] = 0

            grid_mask2d = grid_mask2d.clone()
            a = outputs[grid_mask2d]  # [word的数量,label的数量N]
            b = grid_labels[grid_mask2d]  # [word的数量]
            loss = self.criterion(a, b)
            loss.backward()

            # 对parameters里的所有参数的梯度进行规范化
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            '''
            arg_c: 预测和真实的交集的数量，即预测正确的数量
            arg_p：预测的数量
            arg_set
            '''

            if epoch > 10:
                arg_c, arg_p, arg_r, decode_arguments, arg_span_c, arg_span_p, arg_span_r = utils.decode(
                    outputs.cpu().numpy(), entity_text, sent_length.cpu().numpy())
                total_arg_r += arg_r
                total_arg_p += arg_p
                total_arg_c += arg_c

                total_arg_r_span += arg_span_r
                total_arg_p_span += arg_span_p
                total_arg_c_span += arg_span_c


            '''
            非真实f1，
            '''
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)
            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        a_span_f1, a_span_p, a_span_r = utils.cal_f1(total_arg_c_span, total_arg_p_span, total_arg_r_span)

        a_f1, a_p, a_r = utils.cal_f1(total_arg_c, total_arg_p, total_arg_r)  # 论元角色分类f1

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["argument_span", "占个位"] + ["{:3.4f}".format(x) for x in [a_span_f1, a_span_p, a_span_r]])  # 论元角色分类f1
        table.add_row(["argument", "占个位"] + ["{:3.4f}".format(x) for x in [a_f1, a_p, a_r]])  # 论元角色分类f1
        logger.info("\n{}".format(table))


        return f1

    def eval(self, epoch, data_loader, event_role_dic, config, model, logger, is_test=False):
        self.model.eval()
        pred_result = []
        label_result = []
        total_arg_r = 0
        total_arg_p = 0
        total_arg_c = 0
        total_arg_r_span = 0
        total_arg_p_span = 0
        total_arg_c_span = 0
        with torch.no_grad():
            for i, data_batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
                entity_text = data_batch[-4]
                tri_index = data_batch[-2]
                data_batch = [data_batch[0].cuda(), data_batch[1].cuda(), data_batch[2].cuda(), data_batch[3].cuda(),
                              data_batch[4].cuda(), data_batch[5].cuda(), data_batch[-3].cuda(), data_batch[-1]]

                '''
                bert_inputs:[B, L']
                grid_labels, grid_mask2d, dist_inputs:[B, L, L]
                pieces2word:[B, L, L']
                '''
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, tri_inputs, tri_types = data_batch

                '''
                需要传入模型进行训练的有bert_inputs，grid_mask2d，dist_inputs，pieces2word，sent_length
                '''
                # outputs [B, L, L, num_class] num_class指的是有多少类实体
                outputs = model.forward(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, tri_inputs, tri_index, tri_types, event_role_dic, config)
                grid_mask2d = grid_mask2d.clone()

                '''
                # 处理outputs的约束
                '''
                # for ii, jj in enumerate(tri_types):
                #     roles = event_role_dic[jj]
                #     roles.append(0)
                #     roles.append(1)
                #     array = np.array(range(0, config.vocab.arg_label2id.__len__()))
                #     no_roles = set(array) - set(roles)
                #     for pp in no_roles:
                #         outputs[ii, :, :, pp] = 0

                outputs = torch.argmax(outputs, -1)
                '''
                    arg_c: 预测和真实的交集的数量，即预测正确的数量
                    arg_p：预测的数量
                    arg_set
                '''
                arg_c, arg_p, arg_r, decode_arguments, arg_span_c, arg_span_p, arg_span_r = utils.decode(
                    outputs.cpu().numpy(), entity_text, sent_length.cpu().numpy())
                total_arg_r += arg_r
                total_arg_p += arg_p
                total_arg_c += arg_c

                total_arg_r_span += arg_span_r
                total_arg_p_span += arg_span_p
                total_arg_c_span += arg_span_c

                '''
                非真实f1
                '''
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())


        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        a_span_f1, a_span_p, a_span_r = utils.cal_f1(total_arg_c_span, total_arg_p_span, total_arg_r_span)

        a_f1, a_p, a_r = utils.cal_f1(total_arg_c, total_arg_p, total_arg_r)  # 论元角色分类f1
        title = "EVAL" if not is_test else "TEST"
        table = pt.PrettyTable(["{} {}".format(title, epoch), "F1", "Precision", "Recall"])
        table.add_row(["label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["argument_span"] + ["{:3.4f}".format(x) for x in [a_span_f1, a_span_p, a_span_r]])  # 论元角色分类f1
        table.add_row(["argument"] + ["{:3.4f}".format(x) for x in [a_f1, a_p, a_r]])  # 论元角色分类f1
        logger.info("\n{}".format(table))

        return a_f1



    def predict(self, epoch, data_loader, data, event_role_dic, config, model, logger):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_arg_r = 0
        total_arg_p = 0
        total_arg_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i+config.batch_size]
                entity_text = data_batch[-4]
                tri_index = data_batch[-2]
                data_batch = [data_batch[0].cuda(), data_batch[1].cuda(), data_batch[2].cuda(), data_batch[3].cuda(),
                              data_batch[4].cuda(), data_batch[5].cuda(), data_batch[-3].cuda(), data_batch[-1]]

                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, tri_inputs, tri_types = data_batch

                outputs = model.forward(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length, tri_inputs, tri_index, tri_types, event_role_dic, config)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                # outputs = outputs[grid_mask2d].contiguous().view(-1)

                arg_c, arg_p, arg_r, decode_arguments, arg_span_c, arg_span_p, arg_span_r = utils.decode(
                    outputs.cpu().numpy(), entity_text, sent_length.cpu().numpy())

                for arg_list, sentence in zip(decode_arguments, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "arguments": []}
                    for arg in arg_list:
                        instance["argument"].append({"text": [sentence[x] for x in arg[0]],
                                                   "type": config.vocab.argid_to_label(arg[1])})
                    result.append(instance)

                total_arg_r += arg_r
                total_arg_p += arg_p
                total_arg_c += arg_c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        a_span_f1, a_span_p, a_span_r = utils.cal_f1(total_arg_c_span, total_arg_p_span, total_arg_r_span)
        a_f1, a_p, a_r = utils.cal_f1(total_arg_c, total_arg_p, total_arg_r)  # 论元角色分类f1

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["argument_span"] + ["{:3.4f}".format(x) for x in [a_span_f1, a_span_p, a_span_r]])
        table.add_row(["arguments"] + ["{:3.4f}".format(x) for x in [a_f1, a_p, a_r]])

        logger.info("\n{}".format(table))

        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return a_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def main_model_fzin_by1_3():
    from model_fzin_by1_3 import Model
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/LEVEN.json')
    parser.add_argument('--save_path', type=str, default='./model/3.pkl')
    # parser.add_argument('--save_path', type=str, default='./model/twoConv_nolinear_Newact.pkl')
    parser.add_argument('--predict_path', type=str, default='./result/output.json')
    parser.add_argument('--device', type=int, default=DEVICE)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    config = Config(args)
    logger = utils.get_logger(config.dataset, config.save_path)
    config.logger = logger
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    logger.info("Loading Data")
    datasets, ori_data, tokenizer = data_loader.load_data_bert(config)
    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   # 如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
                   collate_fn=data_loader.collate_fn,
                   # 若shuffle=，则在每个epoch开始的时候，对数据进行重新打乱
                   shuffle=i == 0,
                   # num_workers决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
                   num_workers=8,
                   # drop_last默认为false，即最后一批数据不够batch_size也放入模型正常运行。若为true，则最后一批数据不足量就直接丢弃。
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )
    updates_total = len(datasets[0]) // config.batch_size * config.epochs  # 用来改进scheduler

    # 全局变量 EVENT_ROLE = {}的形成
    event_role_dic = Dic(config)

    model = Model(config, tokenizer)
    model = model.cuda()

    trainer = Trainer(model, config, updates_total)
    best_eval_f1 = 0
    best_test_f1 = 0
    # trainer.model.load_state_dict(torch.load("./model/in_nolinear_Newact7210.pkl"))
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader, event_role_dic, config, model, logger)  # 训练
        eval_f1 = trainer.eval(i, dev_loader, event_role_dic, config, model, logger)  # dev评估
        test_f1 = trainer.eval(i, test_loader, event_role_dic, config, model, logger, is_test=True)  # 测试集评估
        if eval_f1 > best_eval_f1:
            best_eval_f1 = eval_f1
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            trainer.save(config.save_path)
    logger.info("Best DEV F1: {:3.4f}".format(best_eval_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    # trainer.load("./model/in_nolinear_Newact7210.pkl")
    # trainer.predict("Final", test_loader, ori_data[-1], event_role_dic, config, model, logger)
    # trainer.model.load_state_dict(torch.load("./model/wdEn_convPlus.pkl"))
    # for i in range(3):
    #     trainer.other(i, train_loader, event_role_dic, config)
    #     trainer.other(i, dev_loader, event_role_dic, config)
    #     trainer.other(i, test_loader, event_role_dic, config, is_test=True)
    # trainer.predict("Final", test_loader, ori_data[-1])


if __name__ == '__main__':

    main_model_fzin_by1_3()  # 72.10
    main_model_fzin_by1_3()  # 72.10

