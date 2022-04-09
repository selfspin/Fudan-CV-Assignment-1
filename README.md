# Fudan CV Assignment-1

姜俊哲 19307130152

### 函数文件说明

- `standardizeCols.py` : 对数据的标准化操作
- `neuralNetwork.py` : 对传入参数训练神经网络
- `MLPRegularLoss.py` : 返回正则化相关的梯度
- `MLPclassificationLoss.py` : 返回损失函数相关的梯度
- `MLPclassificationPredict.py` : 使用模型对测试数据进行预测
- `find_best_coefficient.py` : 枚举训练参数找到对验证集最好的模型，并保存参数及模型
- `test_finalmodel.py` : 对最终模型在测试集上预测，输出误差

### 模型训练

利用命令使用默认参数（参数查找后最好的参数）训练

```shell
python neuralNetwork.py
```

也可以自己指定参数，例如

```shell
python neuralNetwork.py --layer 100 --lr 0.1 --regular 0.2
```

使用如下命令查看帮助

```shell
python neuralNetwork.py -h
```

训练完毕会同时保存训练集和验证集的error和loss曲线

### 参数查找

可以利用命令在默认参数中查找

```shell
python find_best_coefficient.py
```

您也可以自己选取值，例如

```shell
python find_best_coefficient.py --layer 30 40 50 --lr 1e-3 1e-4 --regular 1e-2 1e-3
```

使用如下命令查看帮助

```shell
python find_best_coefficient.py -h
```

查找完毕会同时保存超参数和最好模型参数

### 模型测试

使用命令测试最终模型

```shell
python test_finalmodel.py
```

