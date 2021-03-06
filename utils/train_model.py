#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime
import cv2

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import torchvision
#from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
# from loc_loss import Localizationloss

from utils.utils import LossRecord, clip_gradient
from eval_utils import get_best_f1_scores, compute_f1, multihot
from config import CLASS_9, CLASS_15
from pytorch_transformers import BertModel, BertConfig, BertTokenizer


import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def get_ce_loss(input_features, onehot_labels):
    input_features = F.softmax(input_features)
    # a = torch.log(input_features)
    sample_loss = -torch.log(input_features) * onehot_labels
    # b = input_features.size(0)  
    loss = sample_loss.sum()/input_features.size(0)
    return loss


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          base_lr,
          exp_lr_scheduler,
          warmup_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluation

    step = 0
    rec_loss = []
    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    # get_ce_loss = nn.CrossEntropyLoss()
    # get_ce_loss = nn.BCEWithLogitsLoss()

    writer = SummaryWriter()
    for epoch in range(start_epoch, epoch_num+1):
        # exp_lr_scheduler.step(epoch)
        warmup_scheduler.step(epoch)
        model.train(True)

        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            model.train(True)

            inputs, labels, img_names = data


            # inputs = Variable(inputs.cuda())
            # labels = Variable(torch.from_numpy(labels).cuda())
        


            labels = np.array(labels)
            labels[labels > 0] = 1
            
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(labels).cuda())

            optimizer.zero_grad()

            outputs = model(inputs)

            loss_f1 = get_ce_loss(outputs[0], labels)
            # loss_f1 = get_ce_loss(outputs[0], labels_15)


            # loss_loc = Localizationloss().forward(model,inputs,img_names,labels)
            loss = loss_f1

            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            # warmup_scheduler.dampen()
            torch.cuda.synchronize()

            print('step: {:-8d} / {:d} loss=ce_loss: {:6.4f} = {:6.4f}'.format(step, train_epoch_step, loss.detach().item(), loss_f1.detach().item()), flush=True)
            rec_loss.append(loss.detach().item())
            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
                print('current exp lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                print('current warmup lr:%s' % warmup_scheduler.get_lr(), flush=True)
                with torch.no_grad():
                    # val set
                    val_score = []
                    val_targets = []
                    val_labels = []
                    val_predict = []
                    for batch_cnt_val, data_val in enumerate(data_loader["val"]):
                        inputs,labels,img_name=data_val
                        inputs = Variable(inputs.cuda())

                        outputs = model(inputs)

                        predict = F.softmax(outputs[0])
                        val_labels.extend(labels)
                        val_predict.extend(-predict.cpu().numpy())

                        for i in range(len(predict)):
                            val_score.append(np.asarray(predict[i].cpu()).tolist())
                            val_targets.append(labels[i])
                    val_score = np.array(val_score)
                    multihot_val = multihot(val_targets, 28)
                    f1 = get_best_f1_scores(multihot_val, val_score)

                    # test set
                    test_score = []
                    test_targets = []
                    for batch_cnt_test, data_test in enumerate(data_loader["test"]):
                        inputs_test, labels_test, img_name_test = data_test
                        inputs_test = Variable(inputs_test.cuda())

                        outputs_test = model(inputs_test)

                        predict_test = F.softmax(outputs_test[0])
                        val_labels.extend(labels_test)
                        val_predict.extend(-predict_test.cpu().numpy())
                        for j in range(len(predict_test)):
                            test_score.append(np.asarray(predict_test[j].cpu()).tolist())                       
                            test_targets.append(labels_test[j])

                    test_score = np.array(test_score)
                    multihot_test = multihot(test_targets, 28)
                    test_micro, test_samples, test_macro, test_none = compute_f1(multihot_test, test_score,
                                                                                 f1["threshold"])

                    print("val_result:")
                    print("macro: " + str(f1['macro']))
                    print("micro: " + str(f1['micro']))
                    print("samples: " + str(f1['samples']))
                    print("none: " + str(f1['none']))
                    print("test_result:")
                    print("macro: " + str(test_macro))
                    print("micro: " + str(test_micro))
                    print("samples: " + str(test_samples))
                    print("none: " + str(test_none))

                    loc = np.argsort(val_predict)  # ??????????????????????????????????????????????????????????????????????????????????????????
                    sum_batch = []
                    for k in range(len(val_labels)):
                        mid_label = val_labels[k]
                        mid_loc = loc[k][:len(mid_label)]
                        mid = 0
                        for w in mid_loc:
                            if w in mid_label:
                                mid = mid + 1
                        mid = mid / len(mid_label)
                        sum_batch.append(mid)
                    sum_all = sum(sum_batch) / len(val_labels)
                    print("test_acc:", sum_all)



                writer.add_scalar('train_loss', train_loss_recorder.get_val(), epoch)  # ?????????
                writer.add_scalar('test_acc', sum_all, epoch)  # ?????????
                writer.add_scalar('val_macro', f1['macro'], epoch)  # ?????????
                writer.add_scalar('val_mirco', f1['micro'], epoch)  # ?????????
                writer.add_scalar('val_sample', f1['samples'], epoch)  # ?????????
                writer.add_scalar('sum_all', f1['macro']+f1['micro']+f1['samples'], epoch)  # ?????????

                save_path = os.path.join(save_dir,'weights_ce+loc_%d_%d.pth' % (epoch, batch_cnt))
                torch.cuda.synchronize()
                torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                torch.cuda.empty_cache()


    writer.close()




