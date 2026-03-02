根据您提供的代码，训练流程可简述如下：

训练流程简述

本实验采用 MindSpore 框架对 LeNet-5 模型进行端到端训练，整体流程分为模型与优化器初始化、训练循环、测试评估三个阶段，具体步骤如下：

1. 模型与训练组件初始化
实例化 LeNet5 网络；
定义损失函数为 SoftmaxCrossEntropyWithLogits（适用于多分类任务）；
选用 Momentum 优化器，学习率设为 0.01，动量系数为 0.9，并传入模型所有可训练参数。

2. 训练循环（train_loop）
利用 mindspore.value_and_grad 自动计算损失对模型参数的梯度；
在每个 batch 中执行前向传播 → 计算损失 → 反向求导 → 参数更新；
每处理 100 个 batch 打印当前 loss 值，用于监控训练过程；
调用 model.set_train() 启用训练模式（如启用 BatchNorm 等训练行为）。

3. 测试评估（test_loop）
每完成一个 epoch 的训练后，在测试集上评估模型性能；
设置 model.set_train(False) 进入推理模式；
遍历整个测试集，计算平均损失和分类准确率；
输出格式如：Accuracy: 98.5%, Avg loss: 0.045678。

4. 训练调度与模型保存
总共训练 100 个 epoch；
每个 epoch 结束后立即保存模型检查点（.ckpt 文件），便于后续加载或断点续训；
训练完成后输出 "Done!" 提示。

该流程结构清晰、模块化程度高，充分利用了 MindSpore 的自动微分和数据 pipeline 能力，确保了训练的高效性与可复现性。