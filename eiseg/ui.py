from eiseg.widget.create import creat_dock, create_button, create_slider, create_text
import sys
import os.path as osp
from enum import Enum
from functools import partial

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QGraphicsView

from eiseg import pjpath, __APPNAME__
import models
from util import MODELS, Instructions
from widget import LineItem, GripItem, AnnotationScene, AnnotationView
from widget.create import *


class Ui_EISeg(object):
    def setupUi(self, MainWindow):
        ## -- 主窗体设置 --
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(1366, 768))
        MainWindow.setWindowTitle(__APPNAME__)
        CentralWidget = QtWidgets.QWidget(MainWindow)
        CentralWidget.setObjectName("CentralWidget")
        MainWindow.setCentralWidget(CentralWidget)
        ## -----
        ## -- 工具栏 --
        toolBar = QtWidgets.QToolBar(self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(toolBar.sizePolicy().hasHeightForWidth())
        toolBar.setSizePolicy(sizePolicy)
        toolBar.setMinimumSize(QtCore.QSize(0, 33))
        toolBar.setMovable(True)
        toolBar.setAllowedAreas(QtCore.Qt.BottomToolBarArea | QtCore.Qt.TopToolBarArea)
        toolBar.setObjectName("toolBar")
        self.toolBar = toolBar
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        ## -----
        ## -- 状态栏 --
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet("QStatusBar::item {border: none;}")
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.addPermanentWidget(
            self.show_logo(osp.join(pjpath, "resource/Paddle.png"))
        )
        ## -----
        ## -- 图形区域 --
        ImageRegion = QtWidgets.QHBoxLayout(CentralWidget)
        ImageRegion.setObjectName("ImageRegion")
        # 滑动区域
        self.scrollArea = QtWidgets.QScrollArea(CentralWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        ImageRegion.addWidget(self.scrollArea)
        # 图形显示
        self.scene = AnnotationScene()
        self.scene.addPixmap(QtGui.QPixmap())
        self.canvas = AnnotationView(self.scene, self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setAutoFillBackground(False)
        self.canvas.setStyleSheet("background-color: White")
        self.canvas.setObjectName("canvas")
        self.scrollArea.setWidget(self.canvas)
        ## -----
        ## -- 工作区 --
        p_create_dock = partial(self.creat_dock, MainWindow)
        p_create_button = partial(self.create_button, CentralWidget)
        # 模型加载
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        ModelRegion = QtWidgets.QVBoxLayout()
        ModelRegion.setObjectName("ModelRegion")
        # labShowSet = self.create_text(CentralWidget, "labShowSet", "模型选择")
        # ModelRegion.addWidget(labShowSet)
        combo = QtWidgets.QComboBox(self)
        combo.addItems([m.__name__ for m in MODELS])
        self.comboModelSelect = combo
        ModelRegion.addWidget(self.comboModelSelect)
        # 网络参数
        self.btnParamsSelect = p_create_button(
            "btnParamsLoad", "加载网络参数", osp.join(pjpath, "resource/Model.png"), "Ctrl+D"
        )
        ModelRegion.addWidget(self.btnParamsSelect)  # 模型选择
        horizontalLayout.addLayout(ModelRegion)
        self.ModelDock = p_create_dock("ModelDock", "模型区", widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.ModelDock)
        # 数据列表
        # TODO: 数据列表加一个搜索功能
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        ListRegion = QtWidgets.QVBoxLayout()
        ListRegion.setObjectName("ListRegion")
        # labFiles = self.create_text(CentralWidget, "labFiles", "数据列表")
        # ListRegion.addWidget(labFiles)
        self.listFiles = QtWidgets.QListWidget(CentralWidget)
        self.listFiles.setObjectName("ListFiles")
        ListRegion.addWidget(self.listFiles)
        # 保存
        self.btnSave = p_create_button(
            "btnSave", "保存", osp.join(pjpath, "resource/Save.png"), "Ctrl+S"
        )
        ListRegion.addWidget(self.btnSave)
        horizontalLayout.addLayout(ListRegion)
        self.DataDock = p_create_dock("DataDock", "数据区", widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.DataDock)
        # 标签列表
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        LabelRegion = QtWidgets.QVBoxLayout()
        LabelRegion.setObjectName("LabelRegion")
        # labelListLab = self.create_text(CentralWidget, "labelListLab", "标签列表")
        # LabelRegion.addWidget(labelListLab)
        self.labelListTable = QtWidgets.QTableWidget(CentralWidget)
        self.labelListTable.horizontalHeader().hide()
        # 铺满
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.labelListTable.verticalHeader().hide()
        self.labelListTable.setColumnWidth(0, 10)
        # self.labelListTable.setMinimumWidth()
        self.labelListTable.setObjectName("labelListTable")
        LabelRegion.addWidget(self.labelListTable)
        self.btnAddClass = p_create_button(
            "btnAddClass", "添加标签", osp.join(pjpath, "resource/Label.png")
        )
        LabelRegion.addWidget(self.btnAddClass)
        horizontalLayout.addLayout(LabelRegion)
        self.LabelDock = p_create_dock("LabelDock", "标签区", widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.LabelDock)
        ## 滑块设置
        # 分割阈值
        p_create_slider = partial(self.create_slider, CentralWidget)
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        ShowSetRegion = QtWidgets.QVBoxLayout()
        ShowSetRegion.setObjectName("ShowSetRegion")
        self.sldThresh, SegShowRegion = p_create_slider(
            "sldThresh", "labThresh", "分割阈值："
        )
        ShowSetRegion.addLayout(SegShowRegion)
        ShowSetRegion.addWidget(self.sldThresh)
        # 透明度
        self.sldOpacity, MaskShowRegion = p_create_slider(
            "sldOpacity", "labOpacity", "标签透明度："
        )
        ShowSetRegion.addLayout(MaskShowRegion)
        ShowSetRegion.addWidget(self.sldOpacity)
        # 点大小
        self.sldClickRadius, PointShowRegion = p_create_slider(
            "sldClickRadius", "labClickRadius", "点击可视化半径：", 3, 10, 1
        )
        ShowSetRegion.addLayout(PointShowRegion)
        ShowSetRegion.addWidget(self.sldClickRadius)
        horizontalLayout.addLayout(ShowSetRegion)
        self.ShowSetDock = p_create_dock("ShowSetDock", "设置区", widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.ShowSetDock)
        ## 专业功能区工作区
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        bandRegion = QtWidgets.QVBoxLayout()
        bandRegion.setObjectName("bandRegion")
        # showWay = create_text(CentralWidget, "showWay", "显示方法")
        # bandRegion.addWidget(showWay)
        # self.rsShow = QtWidgets.QComboBox()
        # self.rsShow.addItems(["原图", "2%线性拉伸"])
        # bandRegion.addWidget(self.rsShow)
        bandSelection = create_text(CentralWidget, "bandSelection", "波段设置")
        bandRegion.addWidget(bandSelection)
        text_list = ["R", "G", "B"]
        self.bandCombos = []
        for txt in text_list:
            lab = create_text(CentralWidget, "band" + txt, txt)
            combo = QtWidgets.QComboBox()
            combo.addItems(["band_1"])
            self.bandCombos.append(combo)
            hbandLayout = QtWidgets.QHBoxLayout()
            hbandLayout.setObjectName("hbandLayout")
            hbandLayout.addWidget(lab)
            hbandLayout.addWidget(combo)
            hbandLayout.setStretch(1, 4)
            bandRegion.addLayout(hbandLayout)
        horizontalLayout.addLayout(bandRegion)
        self.RSDock = p_create_dock("RSDock", "遥感区", widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.RSDock)
        # TODO：添加医疗功能的工作区
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        MIRegion = QtWidgets.QVBoxLayout()
        MIRegion.setObjectName("MIRegion")
        mi_text = create_text(CentralWidget, "bandSelection", "医疗设置")
        MIRegion.addWidget(mi_text)
        horizontalLayout.addLayout(MIRegion)
        self.MIDock = p_create_dock("RSDock", "医疗区", widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.MIDock)
        ## -----
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    ## 创建文本
    def create_text(self, parent, text_name=None, text_text=None):
        return create_text(parent, text_name, text_text)

    ## 创建按钮
    def create_button(self, parent, btn_name, btn_text, ico_path=None, curt=None):
        return create_button(parent, btn_name, btn_text, ico_path, curt)

    ## 创建dock
    def creat_dock(self, parent, name, text, layout):
        return creat_dock(parent, name, text, layout)

    ## 显示Logo
    def show_logo(self, logo_path):
        labLogo = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        labLogo.setSizePolicy(sizePolicy)
        labLogo.setMaximumSize(QtCore.QSize(100, 33))
        labLogo.setPixmap(QtGui.QPixmap(logo_path))
        labLogo.setScaledContents(True)
        labLogo.setObjectName("labLogo")
        return labLogo

    ## 创建滑块区域
    def create_slider(
        self,
        parent,
        sld_name,
        text_name,
        text,
        default_value=50,
        max_value=100,
        text_rate=0.01,
    ):
        return create_slider(
            parent,
            sld_name,
            text_name,
            text,
            default_value,
            max_value,
            text_rate,
        )