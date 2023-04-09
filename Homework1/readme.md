训练步骤<br>
训练时在命令行输入python main.py即可进行训练，可以通过传参对epochs，隐藏层大小，学习率，batch size，正则化系数进行调节。<br>
例如 python main.py --epochs 35 --hidden_size 160 --lr 0.5 --batch_size 64 --L2 0.0001<br>
如果不设置参数，那么将用默认参数进行训练。<br>
训练结束后在picture文件夹下查看图片，在models文件夹下查看保存的模型。<br>
<br>
测试<br>
在test.py中设置想要加载的模型，运行即可得到测试结果。
