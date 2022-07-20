from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import torch
import torch.nn.functional as F
import os
import cv2
import sys
import os.path

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1085, 780)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(0, 10, 661, 711))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(790, 700, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(680, 30, 371, 651))
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(890, 700, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.pushButton.clicked.connect(self.on_upload_img_clicked)
        self.pushButton_2.clicked.connect(self.on_close_button_clicked)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "动漫人物表情识别"))
        self.label.setText(_translate("Dialog", ""))
        self.pushButton.setText(_translate("Dialog", "上传图片"))
        self.label_2.setText(_translate("Dialog", ""))
        self.pushButton_2.setText(_translate("Dialog", "关闭"))

    def on_upload_img_clicked(self):
        # 打开文件对话框
        imgName, imgType = QFileDialog.getOpenFileName(None, '选择图片', '', '*.jpg;;*.png;;AllFile(*.*)')
        if imgName != "":
            detect(imgName, "output.jpg")
        # 获取绝对路径
        url_father = os.path.dirname(os.path.abspath(__file__))

        # 因为styleSheet里正斜杠才管用，我要把反斜杠转化为正斜杠
        url = ""
        for i in url_father:
            if (i == "\\"):
                url = url + "/"
            else:
                url = url + i

        if len(predicts) == 0:
            text = "未识别到人物头像！"
            jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
        else:
            # 合成新的路径并使用
            jpg = QtGui.QPixmap(url+'/output.jpg').scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
            # 设置换行
            self.label_2.setWordWrap(True)
            text = ""
            i = 0
            for predict in predicts:
                i += 1
                text += "face" + str(i) + ": \n"
                j = 0
                for category in categories:
                    text += "{} : {:.2f} %\n".format(str(category), predict[0][j] * 100)
                    j += 1
        self.label_2.setText(text)
        # 清空predicts
        predicts.clear()

    # 按钮单击事件的方法（自定义的槽）
    def on_close_button_clicked(self):
        app = QApplication.instance()
        # 退出应用程序
        app.quit()
# Ensemble模型
class Net1_mix(torch.nn.Module):
    def __init__(self):
        super(Net1_mix, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2)
        self.pooling2 = torch.nn.MaxPool2d(4)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling1(F.relu(self.conv1(x)))
        x = self.pooling2(F.relu(self.conv2(x)))
        x = self.pooling2(F.relu(self.conv3(x)))
        # flatten
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Net2_mix(torch.nn.Module):
    def __init__(self):
        super(Net2_mix, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2)
        self.pooling2 = torch.nn.MaxPool2d(4)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling1(F.relu(self.conv1(x)))
        x = self.pooling2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pooling2(x)
        # flatten
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Net3_mix(torch.nn.Module):
    def __init__(self):
        super(Net3_mix, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2)
        self.pooling2 = torch.nn.MaxPool2d(4)
        self.fc1 = torch.nn.Linear(4096, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.pooling1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pooling2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pooling2(x)
        # flatten
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class NetMix(torch.nn.Module):
    def __init__(self):
        super(NetMix, self).__init__()
        self.net1 = model_net1
        self.net2 = model_net2
        self.net3 = model_net3
        for p in self.parameters():
            p.requires_grad = False  # 预训练模型加载进来后全部设置为不更新参数，然后再后面加层
        self.fc = torch.nn.Linear(4096 * 3, 7)

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net1(x)
        x2 = self.net2(x)
        x3 = self.net3(x)
        # 拼接输入
        x = torch.cat((x1, x2, x3), dim=1)
        # 输出
        x = self.fc(x)
        return x


def detect(filename, output_name):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(24, 24))
    i = 0
    for (x, y, w, h) in faces:
        i += 1
        # 获取脸部识别图像
        image_face = image[y:y + h, x:x + w]
        # 原图标注
        cv2.putText(image, "face" + str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 获取绝对路径
        out_path = os.path.dirname(os.path.abspath(__file__))

        # 不存在文件夹则创建
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if not os.path.exists(out_path + output_name):
            cv2.imwrite(out_path + "/" + output_name, image)

        # 转换图像大小，默认双线性插值
        image_face = cv2.resize(image_face, (128, 128))
        with torch.no_grad():
            image_face = torch.Tensor(image_face)
            model.cpu()
            #             image_face = image_face.to(device)
            image_face = image_face.view(1, 3, 128, 128)
            # 调用模型
            predicts.append(F.softmax(model(image_face)).cpu().numpy().tolist())

if __name__ == "__main__":
    # lbpcascade_animeface.xml位置
    cascade_file = r"D:\lbpcascade_animeface-master\lbpcascade_animeface.xml"
    # 网络参数存放位置参数
    save_path = 'D://jupyter-notebook//ML-Basic-to-improving//Final-Project//model_save/'
    state_dict = torch.load(save_path + "Net_mix_drop_epoch50.pth")
    # 种类和预测
    predicts = []
    categories = ['disgust', 'fear', 'happy', 'surprise', 'sad', 'angry', 'neutral']
    # 声明网络
    model_net1 = Net1_mix()
    model_net2 = Net2_mix()
    model_net3 = Net3_mix()
    model = NetMix()
    model.load_state_dict(state_dict)

    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())