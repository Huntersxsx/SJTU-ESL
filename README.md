## <font face="楷体">任务</font>
本次的课程设计完成的是对28x28像素的QuickDraw图片进行分类任务。  
详细的任务说明见[Kaggle](https://www.kaggle.com/c/statlearning-sjtu-2019/overview)。  
数据下载 [link](https://www.kaggle.com/c/statlearning-sjtu-2019/data)。  
详细的实验步骤见[Report](https://github.com/Huntersxsx/SJTU-ESL/blob/master/统计学习报告.pdf)。  

## <font face="楷体">结果</font>
在[Private Leaderboard](https://www.kaggle.com/c/statlearning-sjtu-2019/leaderboard)和[Public Leaderboard](https://www.kaggle.com/c/statlearning-sjtu-2019/leaderboard)都取得了第四的成绩。  

![](https://github.com/Huntersxsx/SJTU-ESL/blob/master/img/private.png.png)

![](https://github.com/Huntersxsx/SJTU-ESL/blob/master/img/public.png.png)

## <font face="楷体">文件目录</font>
本课程设计的所有代码都放在code压缩包当中。解压后共有CNN_rawdata、CNN_extradata、CNN_moredata、Resnet_rawdata、Resnet_extradata、Resnet_moredata、SVM_rawdata、SVM_extradata、Resnet101_moredata、VGG_moredata和Voting共11个文件夹，分别是对原始数据集和扩展数据集进行训练的SVM、CNN和ResNet18，对数据增强后数据集进行训练的CNN、ResNet18、ResNet101和VGG16，以及投票机制的工程文件夹。'rawdata'结尾的代表是在原始数据集上进行训练的，'extradata'结尾的代表的是加入用预处理的自建数据，在扩充数据集上进行训练的，'moredata'结尾的代表的是在用数据增强后的数据集上进行训练的。


## <font face="楷体">文件说明</font>
上述每个文件夹下都包含了一些用于数据处理和训练的文件。  

##### mean_std.py
用以获取图像数据集的均值和标准差。  

##### datasplit.py
用于划分训练集和验证集，默认按比例7：3进行划分。  

##### dataclean.py
对自建的数据集进行预处理，删去格式不正确、loss值较大的一些噪声图像。获得扩充后的数据集。  

##### createdata.py
采用数据增强，对数据集中每张图片进行平移、旋转、翻转等操作，生成15倍的数据增强后的数据集。  

##### model.py
CNN、ResNet、VGG16等模型的代码文件。  

##### train.py
训练模型的文件。  

##### evaluate.py
对测试集进行预测的文件。  

##### parameters.py
训练时的一些参数，包括比如batch size、初始的learning rate等参数。  

##### utils.py
辅助函数文件，包括保存模型、数据归一化、更改learning rate、生成csv预测结果等函数。  

##### train_state.log
训练过程的记录文件。  

##### Result.csv
对测试集图像预测的结果文件。  



## <font face="楷体">注</font>
&emsp;&emsp;&ensp;由于本课程设计的大部分程序都是在服务器上跑的，所以文件里的文件路径都需要修改后才可以运行，主要就是运行train.py等文件进行模型训练。但是需要先运行mean_std.py、datasplit.py、dataclean.py、createdata.py等数据处理文件后生成新的训练集和验证集才可以运行train.py文件。



