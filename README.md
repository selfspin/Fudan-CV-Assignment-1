# Fudan CV Assignment-1

### 函数文件说明

- `standardizeCols.py` : 对数据的标准化操作
- `neuralNetwork.py` : 对传入参数训练神经网络
- `MLPRegularLoss.py` : 返回正则化相关的梯度
- `MLPclassificationLoss.py` : 返回损失函数相关的梯度
- `MLPclassificationPredict.py` : 使用模型对测试数据进行预测
- `find_best_coefficient.py` : 枚举训练参数找到对验证集最好的模型，并保存参数及模型
- `test_finalmodel.py` : 对最终模型在测试集上预测，输出误差

### 参数查找

可以利用命令

```shell
python find_best_coefficient.py
```

在默认参数中查找

您也可以自己选取值，例如

```shell
python find_best_coefficient.py --layer 10 20 30 40 50 --lr 1e-2 1e-3 1e-4 --regular 1e-2 1e-3 1e-4
```

使用

```shell
python find_best_coefficient.py -h
```

查看帮助
