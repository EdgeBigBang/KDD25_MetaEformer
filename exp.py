from itertools import islice

import numpy as np
import pandas as pd
from torch import nn
from torch.optim import Adam, lr_scheduler
import torch
import time
import os
from data_provider.data_factory import data_provider, get_cloud, get_cloud_zeroshot
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import random

from models.MetaEformer.MetaEformer import MetaEformer
from models.global_utils import train_test_split
from utils.tools import adjust_learning_rate, visual

def get_mape(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs((yTrue - yPred) / yTrue) * 100)


def get_mse(yTrue, yPred, scaler=None):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)
    return np.mean((yTrue - yPred) ** 2)


def get_mae(yTrue, yPred, scaler):
    if scaler:
        yTrue = scaler.inverse_transform(yTrue)
        yPred = scaler.inverse_transform(yPred)

    return np.mean(np.abs(yTrue - yPred))

def acquire_device(args):
    if args.use_gpu:
        device = torch.device('cuda:{}'.format(args.gpu))
        print('Use GPU: cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device

def get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


def test(args):
    folder_path = './test_results/'+ args.task_name + '/'
    device = acquire_device(args)
    args.device = device

    model = torch.load('saved_model/MetaEformer_pro_best_n_app.pt')

    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    criterion = nn.MSELoss().to(device)

    if 'Cloud' in args.dataset:
        train_loader, test_loader, yscaler = get_cloud(args)
    else:
        test_data, test_loader = get_data(args, flag='test')
        yscaler = None

    test_loss = []
    test_mse = []
    test_mae = []
    test_mape = []

    epo_test_losses = []
    epo_mse = []
    epo_mape = []
    epo_mae = []
    test_upper_distance = []
    test_lower_distance = []
    model.eval()
    df = pd.DataFrame()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if 'Cloud' in args.dataset:
                batch_x_static = batch_x_mark[:, 0, 3:].float().to(device).squeeze(1)
            else:
                batch_x_static = None

            if 'Cloud' in args.dataset:
                batch_x_mark = batch_x_mark.float().to(device)[:, :, :3]
            else:
                batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs, mpp = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_static, False)
            f_dim = -1
            outputs = outputs[:, -args.pred_len:, f_dim]
            batch_y = batch_y[:, -args.pred_len:, f_dim]
            loss = criterion(outputs, batch_y)
            epo_test_losses.append(loss.item())
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()      

            if 'Cloud' in args.dataset:
                epo_mse.append(get_mse(outputs, batch_y, yscaler))
                epo_mape.append(get_mape(outputs, batch_y, yscaler))
                epo_mae.append(get_mae(outputs, batch_y, yscaler))
            else:
                epo_mse.append(get_mse(outputs, batch_y, None))
                epo_mape.append(get_mape(outputs, batch_y, None))
                epo_mae.append(get_mae(outputs, batch_y, None))

    test_loss.append(np.mean(epo_test_losses))
    test_mse.append(np.mean(epo_mse))
    test_mape.append(np.mean(epo_mape))
    test_mae.append(np.mean(epo_mae))
    print(f'test loss: {test_loss[-1]}, mse: {test_mse[-1]}, mape: {test_mape[-1]}, mae: {test_mae[-1]}')




def train(args):
    device = acquire_device(args)
    args.device = device

    model = MetaEformer(args)
    model = model.to(device)

    if args.use_multi_gpu and args.use_gpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    if 'Cloud' in args.dataset:
        train_loader, test_loader, yscaler = get_cloud(args)
    else:
        train_data, train_loader = get_data(args, flag='train')
        test_data, test_loader = get_data(args, flag='test')
        yscaler = None

    train_steps = len(train_loader)
    train_loss = []
    test_loss = []
    test_mse = []
    test_mae = []
    test_mape = []

    criterion = nn.MSELoss().to(device)

    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.num_epoches,
                                        max_lr=args.learning_rate)

    min_loss = 1000
    
    # training
    for epoch in range(args.num_epoches):
        iter_count = 0
        epo_train_losses = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if 'Cloud' in args.dataset:
                batch_x_static = batch_x_mark[:, 0, 3:].float().to(device).squeeze(1)
            else:
                batch_x_static = None

            if 'Cloud' in args.dataset:
                batch_x_mark = batch_x_mark.float().to(device)[:, :, :3]
            else:
                batch_x_mark = batch_x_mark.float().to(device)

            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs, mpp = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_static, i % args.mpp_update == 0)


            f_dim = -1
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)

            epo_train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            adjust_learning_rate(optimizer, scheduler, epoch + 1, args, printout=False)
            scheduler.step()

        train_loss.append(np.mean(epo_train_losses))

        epo_test_losses = []
        epo_mse = []
        epo_mape = []
        epo_mae = []
        model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                if 'Cloud' in args.dataset:
                    batch_x_static = batch_x_mark[:, 0, 3:].float().to(device).squeeze(1)
                else:
                    batch_x_static = None

                if 'Cloud' in args.dataset:
                    batch_x_mark = batch_x_mark.float().to(device)[:, :, :3]
                else:
                    batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                outputs, mpp = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_x_static, False)

                f_dim = -1
                outputs = outputs[:, -args.pred_len:, f_dim]
                batch_y = batch_y[:, -args.pred_len:, f_dim]
                loss = criterion(outputs, batch_y)
                epo_test_losses.append(loss.item())
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                epo_mse.append(get_mse(outputs, batch_y, yscaler))
                epo_mape.append(get_mape(outputs, batch_y, yscaler))
                epo_mae.append(get_mae(outputs, batch_y, yscaler))

        test_loss.append(np.mean(epo_test_losses))
        test_mse.append(np.mean(epo_mse))
        test_mape.append(np.mean(epo_mape))
        test_mae.append(np.mean(epo_mae))

        if args.save_model:
            if test_loss[-1] < min_loss:
                best_model = model
                min_loss = test_loss[-1]
                torch.save(model, 'saved_model/MetaEformer_best.pt')

        print(f'epoch {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, '
              f'mse: {test_mse[-1]}, mape: {test_mape[-1]}, mae: {test_mae[-1]}')

    print('best_mse:', np.min(test_mse), 'best_mae', np.min(test_mae))

    return train_loss, test_loss, test_mse, test_mae

