# LDA-
基于LDA主题模型和主题困惑度的文本主题提取系统
## 1 前言
随着人工智能和大数据时代的到来,人类日常生活的方方面面都离不开网络和信息技术，现实世界的文本信息更多的呈现为电子版，文本挖掘也成为信息领域的研究热点和学习重点。用计算机实现海量文本的识别和分析成为研究的热门话题。  
现如今，文本的载体逐渐从纸面向荧屏转移，文本挖掘与文本分析也逐渐成为研究热点，然而，市面上用于文本挖掘与分析的工具和方法却存在着学习门槛高和步骤繁杂的问题。开发基于LDA算法的多文本主题提取软件的目的可以概括如下：第一，为文本分析提供窗口化工具，降低操作门槛；第二，集成文本分词与主题提取功能，减少用户繁琐劳动；第三，加入可视化分析工具，提高提取进度与分析效率。
## 2 模型与参数介绍
隐含狄利克雷分布（Latent Dirichlet Allocation，LDA），是一种主题模型（topic model），它可以将文档集中每篇文档的主题按照概率分布的形式给出。LDA主题模型可以看作是一个生成文档的机器，向机器内输入一个文档，机器产生一个文档。这个机器包括含有两对核心组件，分别是两个迪利克雷分布和两个多项式分布。  

<img src="https://github.com/tomatoyou/LDA-topic-extractor/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/1.png" alt="替代文本" width="700px">

### 2.1 超参数α和β
第一个迪利克雷分布是“文档-主题分布”，它对应一个控制参数α，这个分布可以看作一个多边形（或者是多面体，维度取决于topic的个数），每一个顶点对应一个主题，多面体内含有多个点，对应多个文档，点到某个顶点的距离可以理解为文档对应这个主题的概率，距离越近，概率越大。其次是参数α，当α=1时，对应随机分布；当α<1时，各个点趋向于接近顶点，也就是我们需要的效果；当α>1时，各个点趋向于中心聚拢。  

<img src="https://github.com/tomatoyou/LDA-topic-extractor/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/2.png" alt="替代文本" width="700px">

第二个迪利克雷分布是“主题-单词分布”，它对应的控制参数是β，与第一个迪利克雷分布相似，顶点是单词，内部的点是主题。  
有了上面的两个迪利克雷分布，我们可以产生两个与之共轭的多项式分布，然后通过这两个多项式分布按照贝叶斯网络结构产生一系列的单词并 组成一个文档（这种概率受到两个控制参数α，β的约束）。  

<img src="https://github.com/tomatoyou/LDA-topic-extractor/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/3.png" alt="替代文本" width="700px">

当我们人为修改α、β参数，直到找到一篇与输入文档最相似的一篇，此时的α、β参数就是相对较优的参数。最后，通过α迪利克雷分布可以得到文档对应各个主题的概率分布，通过β迪利克雷分布可以得到各个主题下出现各个单词的概率分布。 
### 2.2 主题划分数
主题划分个数也是模型的一个重要参数，它决定了模型最终将文档划分的类别数。从模型内部来看，划分个数影响着α迪利克雷分布的维度，当选取划分个数为3时，α迪利克雷分布可看作文档集在一个三角形内部的分布情况，当选取划分个数为4时，则是4面体，其维度随着划分个数的增加而增加。在设定此参数时，我们通过计算参数的各种取值对应的模型主题困惑度（perplexity）来确定最优取值（困惑度时对主题模型性能的一个评估指标，困惑度越小，说明模型的拟合（预测）能力越强），比如绘制topics-perplexity曲线，选取曲线的极小值点或者拐点。
## 3 软件说明
软件设计所使用的语言为python语言，主要使用的软件包为PyQt5，开发环境为pycharm和qt designer，并使用Pyinstaller将软件打包。  

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E4%B8%BB%E7%95%8C%E9%9D%A2.png" alt="替代文本" width="700px">

### 3.1 功能介绍
软件包含单文本分词和多文本主题提取两个主要功能，在此基础上，用户可以按照规定文件格式自定义导入用户词典，并导出数据处理结果。

#### 3.1.1 单文本分词
用户在“单文本分词”编辑框中输入将要分析的文本，若要清除编辑内容，可点击“重新输入”按钮清除内容。文本编辑好后点击“词频分析”按钮，即可在右边“分词结果”显示框内查看词频最高的10个词汇。若要完整的词频数据，用户可点击“导出excel”按钮导出词频数据，xlsx文件默认保存至当前文件夹下，用户可自行点击查看。

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E5%8D%95%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D.png" alt="替代文本" width="700px">

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E5%88%86%E8%AF%8D%E7%BB%93%E6%9E%9C.png" alt="替代文本" width="700px">

#### 3.1.2 多文本主题分析
多文本主题分析是该软件的核心功能，用户可以将所要处理数据的xlsx文件（或xls文件）拖入窗口内以获取文件路径，或者点击“文件选择”浏览文件，也可以在编辑框内直接输入路径。路径设置好之后，用户可根据文本数据大小和特点自行设置提取的特征词数、主题个数和主题词数，用户未输入情况下，软件默认提取1000个特征词，归类为5个主题，并展示8个体现主题的关键词。点击“主题提取”按钮，即可在“主题提取结果”内查看各个主题的关键词。点击“计算困惑度”按钮，用户可在窗口右下角显示框内查看到困惑度曲线，并根据曲线调整LDA模型的参数，再次进行主题提取。在已进行主题提取操作后，点击“导出html视图”可将主题提取的详细结果以图形的方式保存至目录，打开目录下的html文件可在浏览器查看各主题的关键词及其主题概率。用户点击“导出excel”按钮即可导出数据。导出数据如图10。

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E4%B8%BB%E9%A2%98%E6%8F%90%E5%8F%96.png" alt="替代文本" width="700px">

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E5%8F%AF%E8%A7%86%E5%8C%96%E7%BB%93%E6%9E%9C.png" alt="替代文本" width="700px">

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E5%9B%B0%E6%83%91%E5%BA%A6%E8%AE%A1%E7%AE%97.png" alt="替代文本" width="700px">

#### 3.1.3 添加用户词典
用户在进行单文本分词或多文本主题提取时，可通过该功能添加用户词典，以避免在分词处理时关键词丢失，如图6。文件导入方式与多文本分析类似，用户可通过拖拽、点击浏览和直接输入三种方式设置用户词典路径，并且，用户在进行文件拖拽时，只能将“txt”，“xls”，“xlsx”三种类型文件拖入，软件会根据文件类型对号入座：若拖入txt文件，则将路径设置为用户词典路径，若拖入xls文件或xlsx文件，则设置为分析文本数据路径。

<img src="https://github.com/tomatoyou/LDA-/blob/main/%E7%95%8C%E9%9D%A2%E6%88%AA%E5%9B%BE/%E6%B7%BB%E5%8A%A0%E7%94%A8%E6%88%B7%E8%AF%8D%E5%85%B8.png" alt="替代文本" width="700px">

## 4 参考资料
[1] https://www.bilibili.com/video/BV123411G7Z9/?spm_id_from=333.337.search-card.all.click&vd_source=da5e0dd60be12895200f4a821f52fb26  
[2]	https://www.bilibili.com/video/BV1cJ411R7bP/?spm_id_from=333.337.search-card.all.click&vd_source=da5e0dd60be12895200f4a821f52fb26  
[3]	https://www.bilibili.com/video/BV1LQ4y1Q7xv/?spm_id_from=333.337.search-card.all.click&vd_source=da5e0dd60be12895200f4a821f52fb26  
[4]	https://blog.csdn.net/qq_39496504/article/details/107125284  
[5]	https://blog.csdn.net/TiffanyRabbit/article/details/76445909  
[6]	https://www.bilibili.com/video/BV1t54y127U8  
[7]	https://www.jianshu.com/p/5c510694c07e  
[8]	https://blog.csdn.net/weixin_43343486/article/details/109255165  
[9]	https://blog.csdn.net/weixin_39676021/article/details/112187210  
[10]	https://mp.weixin.qq.com/s/hMcJtB3Lss1NBalXRTGZlQ  
