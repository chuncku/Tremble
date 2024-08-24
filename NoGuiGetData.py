
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import RawDataReceiver, HWResultReceiver, FeatureMapReceiver
import time
import torch
import torch.nn as nn
import numpy as np

# queue length
QUEUE_LEN = 50 #可調整偵測Frames數值
USE_AI = True
# AI
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")
all_class = ['Background', 'Low', 'Middle', 'High']
all_class_zh = ['無', '低', '中', '高']
class_map = {cls: i for i, cls in enumerate(all_class)}
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2), # img_size/2
        )
    def forward(self, x):
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 32, kernel_size=kernel_size),
            ConvBlock(32, 64, kernel_size=kernel_size),
            ConvBlock(64, 128, kernel_size=kernel_size),
        )
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 4),
        )

    def forward(self, x):
        x = self.feature_extractor(x) # img to feature maps
        x = self.flatten(x) # feature maps -> feature vectors
        x = self.classifier(x) # classification
        return x

model = CNN(kernel_size=3).to(device)
# load model on cpu
model.load_state_dict(torch.load('./model3D.pth', map_location=torch.device('cpu')))
model = model.eval()

def connect():
    connect = ConnectDevice()
    connect.startUp()                       # Connect to the device
    reset = ResetDevice()
    reset.startUp()                         # Reset hardware register

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_120cm")  # Set the setting folder name
    ksp = SettingProc()                 # Object for setting process to setup the Hardware AI and RF before receive data
    ksp.startUp(SettingConfigs)             # Start the setting process
    # ksp.startSetting(SettingConfigs)        # Start the setting process in sub_thread

def startLoop():
    # kgl.ksoclib.switchLogMode(True)
    # R = RawDataReceiver(chirps=32)

    # Receiver for getting Raw data
    R = FeatureMapReceiver(chirps=32)       # Receiver for getting RDI PHD map
    # R = HWResultReceiver()                  # Receiver for getting hardware results (gestures, Axes, exponential)
    # buffer = DataBuffer(100)                # Buffer for saving latest frames of data
    R.trigger(chirps=32)                             # Trigger receiver before getting the data
    time.sleep(0.5)
    print('# ======== Start getting gesture ===========')
    # start timer
    start_time = time.time() # 計時器
    # frames in queue
    frames = []
    while True:                             # loop for getting the data
        res = R.getResults()                # Get data from receiver
        # res: (2, ) res[0]: (32, 32)
        if res is None:
            continue
        frames.append(res[0])  # 新增收集 1個(RDI)Frame
        # print('data = {}'.format(res))          # Print results
        # time.sleep(0.05)
        '''
        Application for the data.
        '''
        if len(frames) >= QUEUE_LEN and USE_AI: # 判斷收集的Frames數值是否足夠
            # end timer
            end_time = time.time()
            print(f"Time: {end_time - start_time}")
            # frames_tensor = torch.tensor(frames[-200:], dtype=torch.float32) # (slices, H, W)
            frames_tensor = torch.from_numpy(np.array(frames[-QUEUE_LEN:])).float() # 原本Frame list shape (slices, H, W)
        
            # min max normalize
            frames_tensor = (frames_tensor - frames_tensor.min()) / (frames_tensor.max() - frames_tensor.min())
            # permute to (H, W, slices)  
            frames_tensor = frames_tensor.permute(1, 2, 0)
            # add channel dimension and batch dimension
            frames_tensor = frames_tensor.unsqueeze(0).unsqueeze(0) # (B, C, H, W, slices) 擴增維度 Batchsize=1 Channel=1
            with torch.no_grad():
                frames_tensor = frames_tensor.to(device)
                pred = model(frames_tensor).squeeze(0) # 模型推論
                pred = torch.argmax(pred).cpu().item() # 取出最大值的類別
                print(f"Predict: {all_class[pred]}")
                print(f"AI 評估顫抖等級: {all_class_zh[pred]}")
            # clear frames and time
            frames = [] # 預測結束 清空收集的Frames
            start_time = end_time
            
            
            
def main():
    kgl.setLib()

    # kgl.ksoclib.switchLogMode(True)

    connect()                               # First you have to connect to the device

    startSetting()                         # Second you have to set the setting configs

    startLoop()                             # Last you can continue to get the data in the loop

if __name__ == '__main__':
    main()
