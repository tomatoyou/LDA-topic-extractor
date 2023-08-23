import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QGraphicsScene
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
import jieba
import jieba.posseg as psg
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import numpy as np
import matplotlib.pyplot as plt

stopword = {"这个", "这么", "1", "2", "3", "4", "5", "6", "7", "8"}    # 内置停用词，可自行添加

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(878, 791)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.input_text = QtWidgets.QPlainTextEdit(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.input_text.setFont(font)
        self.input_text.setObjectName("input_text")
        self.verticalLayout.addWidget(self.input_text)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.clear_Button = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.clear_Button.setFont(font)
        self.clear_Button.setObjectName("clear_Button")
        self.horizontalLayout.addWidget(self.clear_Button)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.cipin_Button = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.cipin_Button.setFont(font)
        self.cipin_Button.setObjectName("cipin_Button")
        self.horizontalLayout.addWidget(self.cipin_Button)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.export_word = QtWidgets.QPushButton(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.export_word.setFont(font)
        self.export_word.setObjectName("export_word")
        self.horizontalLayout.addWidget(self.export_word)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 2)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 2)
        self.horizontalLayout.setStretch(4, 1)
        self.horizontalLayout.setStretch(5, 2)
        self.horizontalLayout.setStretch(6, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_5.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.file_path_dir = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.file_path_dir.setFont(font)
        self.file_path_dir.setObjectName("file_path_dir")
        self.horizontalLayout_3.addWidget(self.file_path_dir)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.file_choose_dir = QtWidgets.QPushButton(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.file_choose_dir.setFont(font)
        self.file_choose_dir.setObjectName("file_choose_dir")
        self.horizontalLayout_2.addWidget(self.file_choose_dir)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.file_open = QtWidgets.QPushButton(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.file_open.setFont(font)
        self.file_open.setObjectName("file_open")
        self.horizontalLayout_2.addWidget(self.file_open)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem6)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_4.addLayout(self.verticalLayout_3)
        self.verticalLayout_5.addWidget(self.groupBox_2)
        self.horizontalLayout_4.addLayout(self.verticalLayout_5)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_3)
        font = QtGui.QFont()
        font.setFamily("华文楷体")
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_6.addWidget(self.textBrowser)
        self.horizontalLayout_4.addWidget(self.groupBox_3)
        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 1)
        self.verticalLayout_11.addLayout(self.horizontalLayout_4)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        self.groupBox_4.setFont(font)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_7.addWidget(self.label_2)
        self.file_path_data = QtWidgets.QLineEdit(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.file_path_data.setFont(font)
        self.file_path_data.setObjectName("file_path_data")
        self.horizontalLayout_7.addWidget(self.file_path_data)
        self.file_choose_data = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.file_choose_data.setFont(font)
        self.file_choose_data.setObjectName("file_choose_data")
        self.horizontalLayout_7.addWidget(self.file_choose_data)
        self.verticalLayout_7.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.spinBox_feature = QtWidgets.QSpinBox(self.groupBox_4)
        self.spinBox_feature.setMinimum(100)
        self.spinBox_feature.setMaximum(2000)
        self.spinBox_feature.setProperty("value", 1000)
        self.spinBox_feature.setObjectName("spinBox_feature")
        self.horizontalLayout_6.addWidget(self.spinBox_feature)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem7)
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_6.addWidget(self.label_4)
        self.spinBox_topic = QtWidgets.QSpinBox(self.groupBox_4)
        self.spinBox_topic.setMinimum(1)
        self.spinBox_topic.setMaximum(10)
        self.spinBox_topic.setProperty("value", 5)
        self.spinBox_topic.setObjectName("spinBox_topic")
        self.horizontalLayout_6.addWidget(self.spinBox_topic)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem8)
        self.label_5 = QtWidgets.QLabel(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.spinBox_n_topword = QtWidgets.QSpinBox(self.groupBox_4)
        self.spinBox_n_topword.setMinimum(5)
        self.spinBox_n_topword.setMaximum(25)
        self.spinBox_n_topword.setProperty("value", 8)
        self.spinBox_n_topword.setObjectName("spinBox_n_topword")
        self.horizontalLayout_6.addWidget(self.spinBox_n_topword)
        self.verticalLayout_7.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem9)
        self.lda = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.lda.setFont(font)
        self.lda.setObjectName("lda")
        self.horizontalLayout_5.addWidget(self.lda)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem10)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem11)
        self.vis_Button = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.vis_Button.setFont(font)
        self.vis_Button.setObjectName("vis_Button")
        self.horizontalLayout_5.addWidget(self.vis_Button)
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem12)
        self.export_topic = QtWidgets.QPushButton(self.groupBox_4)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.export_topic.setFont(font)
        self.export_topic.setObjectName("export_topic")
        self.horizontalLayout_5.addWidget(self.export_topic)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem13)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        self.verticalLayout_8.addLayout(self.verticalLayout_7)
        self.verticalLayout_11.addWidget(self.groupBox_4)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox_5)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.verticalLayout_9.addWidget(self.textBrowser_2)
        self.horizontalLayout_8.addWidget(self.groupBox_5)
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("隶书")
        font.setPointSize(12)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.groupBox_6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.graphicsView = QtWidgets.QGraphicsView(self.groupBox_6)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout_10.addWidget(self.graphicsView)
        self.horizontalLayout_8.addWidget(self.groupBox_6)
        self.horizontalLayout_8.setStretch(0, 3)
        self.horizontalLayout_8.setStretch(1, 2)
        self.verticalLayout_11.addLayout(self.horizontalLayout_8)
        self.verticalLayout_11.setStretch(0, 5)
        self.verticalLayout_11.setStretch(2, 4)
        self.verticalLayout_12.addLayout(self.verticalLayout_11)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 878, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "文本主题提取"))
        self.groupBox.setTitle(_translate("MainWindow", "单文本分词"))
        self.input_text.setPlaceholderText(_translate("MainWindow", "请输入文本"))
        self.clear_Button.setText(_translate("MainWindow", "重新输入"))
        self.cipin_Button.setText(_translate("MainWindow", "词频分析"))
        self.export_word.setText(_translate("MainWindow", "导出excel"))
        self.groupBox_2.setTitle(_translate("MainWindow", "用户词典"))
        self.label.setText(_translate("MainWindow", "文件路径"))
        self.file_path_dir.setPlaceholderText(_translate("MainWindow", "输入路径或拖拽文件至窗口内"))
        self.file_choose_dir.setText(_translate("MainWindow", "选择文件"))
        self.file_open.setText(_translate("MainWindow", "添加"))
        self.groupBox_3.setTitle(_translate("MainWindow", "分词结果"))
        self.groupBox_4.setTitle(_translate("MainWindow", "多文本主题提取"))
        self.label_2.setText(_translate("MainWindow", "文件路径"))
        self.file_path_data.setPlaceholderText(_translate("MainWindow", "输入路径或拖拽文件至窗口内"))
        self.file_choose_data.setText(_translate("MainWindow", "选择文件"))
        self.label_3.setText(_translate("MainWindow", "特征词数"))
        self.label_4.setText(_translate("MainWindow", "主题个数"))
        self.label_5.setText(_translate("MainWindow", "主题词数"))
        self.lda.setText(_translate("MainWindow", "主题提取"))
        self.pushButton.setText(_translate("MainWindow", "计算困惑度"))
        self.vis_Button.setText(_translate("MainWindow", "导出html视图"))
        self.export_topic.setText(_translate("MainWindow", "导出excel"))
        self.groupBox_5.setTitle(_translate("MainWindow", "主题提取结果"))
        self.groupBox_6.setTitle(_translate("MainWindow", "主题困惑度"))

class TableWin(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setAcceptDrops(True)  # 允许文件拖拽
        self.listener()     # 初始化监听函数
        self.path_dir = ''     # 用户词典路径
        self.path_data = ''    # 多文本分析的文本路径
        self.data = [[]]    # 单文本分词数据
        self.data_topic = [[]]   # 多文本主题提取结果数据
        self.fenci_flag = 0   # 分词标志，判断是否完成“词频分析”
        self.LDA_flag = 0   # LDA标志，判断是否完成“主题提取”
        self.tword = []   # 各文本主题词数据
        self.lda = None
        self.tf = None
        self.tf_vectorizer = None

    def listener(self): #监听函数
        self.clear_Button.clicked.connect(self.clear)  # 重新输入
        self.cipin_Button.clicked.connect(self.fenci)  # 词频分析
        self.export_word.clicked.connect(self.word_export)  # 导出excel(单文本)
        self.file_open.clicked.connect(self.open_file)  # 打开
        self.file_choose_dir.clicked.connect(self.choice_dir)  # 选择文件(用户词典)
        self.file_choose_data.clicked.connect(self.choice_data)  # 选择文件(文本数据)
        self.lda.clicked.connect(self.lda_analysis)  # 主题提取
        self.pushButton.clicked.connect(self.get_perplexity)  # 计算困惑度
        self.vis_Button.clicked.connect(self.topic_vis)  # 导出html
        self.export_topic.clicked.connect(self.topic_export)  # 导出excel(多文本)

    def clear(self):
        self.input_text.clear()

    def fenci(self):
        text = self.input_text.toPlainText()
        text_lines = text.split('\n')
        if self.path_dir:
            jieba.load_userdict(self.path_dir)
        flag_list = ['n', 'nz', 'vn']
        counts = {}
        for line in text_lines:
            line_seg = psg.cut(line)
            for word_flag in line_seg:
                word = re.sub("[^\u4e00-\u9fa5]", "", word_flag.word)
                if word_flag.flag in flag_list and len(word) > 1 and word not in stopword:
                    counts[word] = counts.get(word, 0) + 1
        word_freq = pd.DataFrame({'word': list(counts.keys()), 'freq': list(counts.values())})
        word_freq = word_freq.sort_values(by='freq', ascending=False)
        l = 10
        if len(word_freq) < l:
            l = len(word_freq)
        self.data = word_freq[['word', 'freq']]
        textout = []
        for i in range(l):
            textout.append(str(self.data.iloc[i, 0]) + '-----' + str(self.data.iloc[i, 1]))
        cipin = '关键词-----词频' + '\n' + '********************'
        for i in range(l):
            cipin = cipin + '\n' + str(textout[i])
        self.textBrowser.setText(cipin)
        self.fenci_flag = 1

    def word_export(self):
        if self.fenci_flag == 0:
            QMessageBox.warning(self, '警告', '请先进行词频分析!')
        else:
            self.data.to_excel("text_cut.xlsx", index=False)
            QMessageBox.warning(self, '提示', '分词数据已保存!')

    def chinese_word_cut(self, mytext):   # 对多个文本进行分词处理
        if self.path_dir:
            jieba.load_userdict(self.path_dir)
        jieba.initialize()
        stopword_list = stopword
        stop_list = []
        flag_list = ['n', 'nz', 'vn']
        for line in stopword_list:
            line = re.sub(u'\n|\\r', '', line)
            stop_list.append(line)
        word_list = []
        seg_list = psg.cut(mytext)
        for seg_word in seg_list:
            word = re.sub(u'[^\u4e00-\u9fa5]', '', seg_word.word)
            find = 0
            for stop_word in stop_list:
                if stop_word == word or len(word) < 2:
                    find = 1
                    break
            if find == 0 and seg_word.flag in flag_list:
                word_list.append(word)
        return (" ").join(word_list)

    def get_top_words(self, model, feature_names, n_top_words):   # 通过LDA获取关键词
        topic_show = ''
        for topic_idx, topic in enumerate(model.components_):
            topic_show = topic_show + "Topic #%d:" % topic_idx + '\n'
            topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            self.tword.append(topic_w)
            topic_show = topic_show + topic_w + '\n'
        self.textBrowser_2.setText(topic_show)

    def lda_analysis(self):
        file_format = self.path_data.split('.')[-1]
        if self.path_data and (file_format == 'xls' or file_format == 'xlsx'):
            self.data_topic = pd.read_excel(self.path_data)
            self.data_topic['content_cutted'] = self.data_topic.content.apply(self.chinese_word_cut)
            n_features = self.spinBox_feature.value()  # 提取1000个特征词语
            self.tf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                            max_features=n_features,
                                            stop_words='english',
                                            max_df=0.5,
                                            min_df=10)
            self.tf = self.tf_vectorizer.fit_transform(self.data_topic.content_cutted)
            n_topics = self.spinBox_topic.value()
            self.lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                            learning_method='batch',
                                            learning_offset=50,
                                            doc_topic_prior=0.1,
                                            topic_word_prior=0.01,
                                            random_state=0)
            self.lda.fit(self.tf)
            n_top_words = self.spinBox_n_topword.value()
            tf_feature_names = self.tf_vectorizer.get_feature_names()
            self.get_top_words(self.lda, tf_feature_names, n_top_words)
            self.LDA_flag = 1
        else:
            QMessageBox.warning(self, '警告', '未导入处理文件或文件格式不正确!')

    def get_perplexity(self):
        if self.LDA_flag == 1:
            plexs = []
            scores = []
            n_max_topics = 16
            for i in range(1, n_max_topics):
                print(i)
                lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                                learning_method='batch',
                                                learning_offset=50, random_state=0)
                lda.fit(self.tf)
                plexs.append(lda.perplexity(self.tf))
                scores.append(lda.score(self.tf))
            n_t = 15  # 区间最右侧的值。注意：不能大于n_max_topics
            x = list(range(1, n_t + 1))
            plt.plot(x, plexs[0:n_t])
            plt.xlabel("number of topics")
            plt.ylabel("perplexity")
            plt.savefig('perplexity.png')
            img = cv2.imread('perplexity.png')
            cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 把opencv 默认BGR转为通用的RGB
            y, x = img.shape[:-1]
            frame = QImage(cvimg, x, y, QImage.Format_RGB888)
            scene = QGraphicsScene()  # 创建画布
            scene.clear()  # 先清空上次的残留
            pix = QPixmap.fromImage(frame)
            scene.addPixmap(pix)

            self.graphicsView.setScene(scene)  # 把画布添加到窗口
            self.graphicsView.show()
        else:
            QMessageBox.warning(self, '警告', '请先提取主题!')

    def topic_vis(self):
        if self.LDA_flag == 1:
            pic = pyLDAvis.sklearn.prepare(self.lda, self.tf, self.tf_vectorizer)
            pyLDAvis.save_html(pic, 'lda_pass' + str(self.spinBox_topic.value()) + '.html')
            QMessageBox.warning(self, '提示', '视图已导出!')
        else:
            QMessageBox.warning(self, '警告', '请先提取主题！')

    def topic_export(self):
        if self.LDA_flag == 1:
            topics = self.lda.transform(self.tf)
            topic = []
            for t in topics:
                topic.append("Topic #" + str(list(t).index(np.max(t))))
            self.data_topic['概率最大的主题序号'] = topic
            self.data_topic['每个主题对应概率'] = list(topics)
            self.data_topic.to_excel("data_topic.xlsx", index=False)
            QMessageBox.warning(self,'提示', '主题提取数据已保存!')
        else:
            QMessageBox.warning(self, '警告', '请先提取主题!')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            try:
                event.setDropAction(Qt.CopyAction)
            except Exception as e:
                print(e)
            event.accept()
        else:
            event.ignore()
    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            file_path = links[0]  # 获取文件绝对路径
            file_format = file_path.split('.')[-1]
            if file_format == 'txt':
                self.path_dir = file_path
                self.file_path_dir.setText(file_path)
            if file_format == 'xls' or file_format == 'xlsx':
                self.path_data = file_path
                self.file_path_data.setText(file_path)
        else:
            event.ignore()

    def choice_dir(self):
        now_path = os.getcwd()  # 获取当前路径
        choice_file_path = QFileDialog.getOpenFileName(self, '选择文件', now_path)
        if choice_file_path[0]:
            self.path_dir = choice_file_path[0]
            self.file_path_dir.setText(self.path_dir)

    def choice_data(self):
        now_path = os.getcwd()  # 获取当前路径
        choice_file_path = QFileDialog.getOpenFileName(self, '选择文件', now_path)
        if choice_file_path[0]:
            self.path_data = choice_file_path[0]
            self.file_path_data.setText(self.path_data)

    def open_file(self):
        if self.path_dir:
            file_format = self.path_dir.split('.')[-1]
            if file_format == 'txt':
                QMessageBox.warning(self, '提示', '用户词典添加成功!')
            else:
                self.path_dir = ''
                QMessageBox.warning(self, '警告', '请选择txt文件!')
        else:
            QMessageBox.warning(self, '警告', '请选择文件!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    table_win = TableWin()
    table_win.show()
    sys.exit(app.exec_())