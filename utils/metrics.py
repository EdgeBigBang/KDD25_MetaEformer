import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def boundary_distance(pred, true, scaler):
    if scaler:
        pred = scaler.inverse_transform(pred)
        true = scaler.inverse_transform(true)
    # 计算超过上界的距离
    upper_distance = np.maximum(pred - true, 0)
    # 计算低于下界的距离
    lower_distance = np.maximum(true - pred, 0)
    # 计算平均值
    return np.mean(upper_distance), np.mean(lower_distance)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    upper_distance, lower_distance = boundary_distance(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr, upper_distance, lower_distance
