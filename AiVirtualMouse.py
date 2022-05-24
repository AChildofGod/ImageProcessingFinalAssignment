# coding: UTF-8
# @Time    : 2022/5/12--22:23
# @Author  : child_lyj
# @File    : AiVirtualMouse.py
# @Software: PyCharm
# @Project : GestureMouse
# @Description:获取屏幕分辨率:wmic path Win32_VideoController get CurrentHorizontalResolution,CurrentVerticalResolution

import time
import cv2
from cvzone.HandTrackingModule import HandDetector  # 手部关键点检测方法
import pyautogui  # autopy太垃圾替换成pyautogui
import numpy

# (1) 导出视频数据
screen_width, screen_height = pyautogui.size()  # 获取电脑屏幕的宽和高:(1920, 1080)
# screen_width, screen_height = int(screen_width), int(screen_height)  # 将电脑屏幕宽高数据转化为整数

camera_width, camera_height = 1280, 720  # 摄像头窗口的宽和高，本人使用笔记本电脑最大支持的相机尺寸
# 左上角和有下角是相对于摄像头的坐标不是相对于屏幕的
top_left_corner, bottom_right_corner = (50, 50), (camera_width - 50, camera_height - 50)  # 手势鼠标的移动范围(左上角坐标)(右下角坐标)
cap = cv2.VideoCapture(0)  # 创建摄像头对象，若使用笔记本自带摄像头则编号为0  若使用外接摄像头 则更改为1或其他编号
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)  # 设置显示框的宽度1280，在视频流的帧的宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)  # 设置显示框的高度720，在视频流的帧的高度

"""
1） cvzone.HandTrackingModule.HandDetector()   手部关键点检测方法
参数：
mode： 默认为 False，将输入图像视为视频流。它将尝试在第一个输入图像中检测手，并在成功检测后进一步定位手的坐标。在随后的图像中，一旦检测到所有 maxHands 手并定位了相应的手的坐标，它就会跟踪这些坐标，而不会调用另一个检测，直到它失去对任何一只手的跟踪。这减少了延迟，非常适合处理视频帧。如果设置为 True，则在每个输入图像上运行手部检测，用于处理一批静态的、可能不相关的图像。
maxHands： 最多检测几只手，默认为 2
detectionCon： 手部检测模型的最小置信值（0-1之间），超过阈值则检测成功。默认为 0.5
minTrackingCon： 坐标跟踪模型的最小置信值 (0-1之间)，用于将手部坐标视为成功跟踪，不成功则在下一个输入图像上自动调用手部检测。将其设置为更高的值可以提高解决方案的稳健性，但代价是更高的延迟。如果 mode 为 True，则忽略这个参数，手部检测将在每个图像上运行。默认为 0.5
"""
# （2）接收手部检测方法: 使用mediapipe库查找手。以像素格式导出地标。增加了一些额外的功能，比如查找竖起的手指数或两个手指之间的距离。
# 还提供找到的手的边界框信息。
detector = HandDetector(mode=False,  # 视频流图像
                        maxHands=1,  # 最多检测一只手
                        detectionCon=0.8,  # 最小检测置信度
                        minTrackCon=0.5)  # 最小跟踪置信度

p_loc_x, p_loc_y = pyautogui.position()  # 保存上一帧时的鼠标所在位置

cv2.namedWindow("cap", cv2.WINDOW_NORMAL)  # 新建一个名为"cap”的窗口用于显示图片
# 设置窗口尺寸为本笔记本支持的最大尺寸，只能使用整数，不能使用浮点数,会放大摄像头的显示尺寸
cv2.moveWindow("cap", 0, 0)  # 设置窗口的显示位置
cv2.resizeWindow("cap", screen_width, screen_height)
while True:

    start_time = time.time()  # 设置第一帧开始处理的起始时间
    smooth = 1  # 自定义平滑系数，让鼠标移动平缓一些，系数越大越平缓

    # (3)处理每一帧图像
    # 当我们使用OpenCV的方法去读取视频的时候，经常使用的就是 VideoCapture.read() 函数,
    # 该函数结合VideoCapture::grab()和VideoCapture::retrieve()，
    # 用于捕获、解码和返回下一个视频帧这是一个最方便的函数对于读取视频文件或者捕获数据从解码和返回刚刚捕获的帧，
    # 假如没有视频帧被捕获（相机没有连接或者视频文件中没有更多的帧）将返回False。

    # 打开摄像头或者视频文件(只是打开摄像头，并不会显示摄像头拍到的显示图像img.imshow才会显示图像)，图片是否成功接收、img帧图像
    is_success, img = cap.read()  # 该方法返回一个元组，第0个元素表示是否捕捉到了视频数据，第二个参数表示解码后的图像数据

    # 翻转图像，使自身和摄像头中的自己呈镜像关系
    img = cv2.flip(img, flipCode=1)  # 1代表水平翻转，0代表竖直翻转

    # 在图像窗口上创建一个矩形框，在该区域内移动鼠标， pt1:左上角，pt2:右下角，color:颜色，thickness翻译为：厚度、粗细
    cv2.rectangle(img, pt1=top_left_corner, pt2=bottom_right_corner, color=(0, 255, 255), thickness=5)

    # （4）手部关键点检测方法
    """ cvzone.HandTrackingModule.HandDetector.findHands()   找到手部关键点并绘图
    参数：
    img： 需要检测关键点的帧图像，格式为BGR
    draw： 是否需要在原图像上绘制关键点及识别框
    flipType： 图像是否需要翻转，当视频图像和我们自己不是镜像关系时，设为True就可以了
    
    返回值：
    hands： 检测到的手部信息，包含：21个关键点坐标，检测框坐标及宽高，检测框中心坐标，检测出是哪一只手。
    img： 返回绘制了关键点及连线后的图像
    """
    # 传入每帧图像， 返回一个元祖，第一个元素：返回手部关键点的坐标信息(字典构成的列表hands)；第二个元素：制关键点后的图像img
    # 需要将手放到摄像头的框内返回的hands才会有信息，否则hands为空值，有几只手hands长度就为几，要注意HandDetector中maxHands参数的设置
    hands, img = detector.findHands(img, flipType=False)  # # 检测手势并画上骨架信息，上面翻转过了，这里就不用再翻转了
    # print(hands)
    # print(len(hands[0]['lmList']), '\n', hands[0]['lmList'])  # 输出21个手部关键点信息
    # 如果能检测到手部，并且是右手就可以进行鼠标点击和拖拽的操作
    if hands and hands[0]['type'] == 'Right':
        # 获取手部信息hands中的21个关键点信息
        lmList = hands[0]['lmList']  # hands是由N个字典组成的列表，其中每个字典包括每只手的关键点信息

        # 获取大拇指指尖、食指指根、食指指尖坐标和中指指尖坐标
        thumb_tip_x, thumb_tip_y, thumb_tip_z = lmList[4]
        index_finger_mcp_x, index_finger_mcp_y, index_finger_mcp_z = lmList[5]
        index_finger_pip_x, index_finger_pip_y, index_finger_pip_z = lmList[6]  # 6号关节的位置
        pinky_tip_x, pinky_tip_y, pinky_tip_z = lmList[20]
        x1, y1, z1 = lmList[8]  # 食指指尖的索引号为8
        x2, y2, z2 = lmList[12]  # 中指指尖的索引号为12
        # print(x1, y1, z1, '\n', x2, y2, z2)
        # (5)检查哪个手指是朝上的
        fingers = detector.fingersUp(hands[0])  # 检测hands中每根手指的朝向
        distance, info, img = detector.findDistance((index_finger_mcp_x, index_finger_mcp_y),
                                                    (thumb_tip_x, thumb_tip_y),
                                                    img)
        # 返回的是两点之间距离，两点之间的直线信息，图像的矩阵形式
        # print(distance)
        # print(fingers)  # 返回 [0,1,1,0,0] 代表 只有食指和中指竖起

        # 如果食指和中指都竖起、无名指和小指都弯曲，就认为是移动鼠标
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0 and distance >= 50:
            # 开始移动鼠标，在在食指指尖画一个圆，看得鞥更清晰一些
            cv2.circle(img=img, center=(x1, y1), radius=15, color=(255, 255, 0), thickness=cv2.FILLED)  # 颜色填充整个圆

            # (6)确定鼠标的移动范围
            # 将食指的移动范围从预制的窗口范围，映射到电脑屏幕范围
            x3 = numpy.interp(x1, (top_left_corner[0], bottom_right_corner[0]), (0, screen_width))  # 将食指的x轴进行映射
            y3 = numpy.interp(y1, (top_left_corner[1], bottom_right_corner[1]), (0, screen_height))  # 将食指的y轴进行映射

            # 平滑，使手指在移动鼠标时，鼠标箭头不会一直晃动
            c_loc_x = p_loc_x + (x3 - p_loc_x) / smooth  # 当前鼠标所在的位置坐标x
            c_loc_y = p_loc_y + (y3 - p_loc_y) / smooth  # 当前鼠标所在的位置坐标y
            # (8)移动鼠标
            pyautogui.moveTo(c_loc_x, c_loc_y)  # 给出鼠标移动位置坐标
            # 更新前一帧鼠标所在的位置坐标，将当前帧鼠标所在位置，变成下一帧的鼠标所在的前一阵位置
            p_loc_x, p_loc_y = c_loc_x, c_loc_y

        # 只要食指指尖竖起且中指弯下，就去判断4号点位和5号点位之间的距离，距离小于50，就认为是鼠标左键点击操作
        if fingers[1] == 1 and fingers[2] == 0 and distance < 50:  # 食指竖起，中指弯下
            # right被定义成常量autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT,down=True)或者autopy.mouse.toggle(autopy.mouse.Button.RIGHT,True)即可
            # 当4号点位和5号点位之间的距离小于50(像素距离)就认为是点击鼠标
            # 在食指尖画个绿色的圆，表示点击鼠标
            cv2.circle(img=img, center=(x1, y1), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            # autopy.mouse.toggle(button=autopy.mouse.Button.LEFT, down=True)
            # pyautogui.mouseDown
            # pyautogui.click(x=moveToX, y=moveToY, clicks=num_of_clicks, interval=secs_between_clicks, button='left')
            # pyautogui.click(clicks=1, interval=0.5, button=pyautogui.LEFT)
            pyautogui.leftClick(interval=0.5)  # 鼠标左键点击操作

        # 只要食指指尖弯下且中指之指尖竖起，就去判断4号点位和5号点位之间的距离，距离小于50，就认为是鼠标右键点击操作
        if fingers[1] == 0 and fingers[2] == 1 and distance < 50:  # 食指竖起，中指弯下
            # right被定义成常量autopy.mouse.toggle(button=autopy.mouse.Button.RIGHT,down=True)或者autopy.mouse.toggle(autopy.mouse.Button.RIGHT,True)即可
            # 当4号点位和5号点位之间的距离小于50(像素距离)就认为是点击鼠标
            # 在中指尖画个绿色的圆，表示点击鼠标
            cv2.circle(img=img, center=(x2, y2), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            pyautogui.rightClick(interval=0.5)  # 鼠标左键点击操作

        # 只要食指指尖竖起且中指指尖竖起，就去判断4号点位和5号点位之间的距离，距离小于50，就认为是鼠标双击操作
        if fingers[1] == 1 and fingers[2] == 1 and distance < 50:  # 食指竖起，中指弯下
            # 当4号点位和5号点位之间的距离小于50(像素距离)就认为是点击鼠标
            # 在食指尖和中指尖画个绿色的圆，表示点击鼠标
            cv2.circle(img=img, center=(x1, y1), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            cv2.circle(img=img, center=(x2, y2), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            pyautogui.doubleClick(interval=0.5)  # 鼠标双击操作
        # 如果小拇指竖起，且食指、中指、无名指都弯曲，就认定是鼠标拖拽功能
        if fingers[4] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
            # 在食指尖和中指尖画个绿色的圆，表示点击鼠标
            cv2.circle(img=img, center=(pinky_tip_x, pinky_tip_y), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            x3 = numpy.interp(pinky_tip_x, (top_left_corner[0], bottom_right_corner[0]),
                              (0, screen_width))  # 将小拇指的x轴进行映射
            y3 = numpy.interp(pinky_tip_y, (top_left_corner[1], bottom_right_corner[1]),
                              (0, screen_height))  # 将小拇指的y轴进行映射

            # 平滑，使手指在移动鼠标时，鼠标箭头不会一直晃动
            c_loc_x = p_loc_x + (x3 - p_loc_x) / smooth  # 当前鼠标所在的位置坐标x
            c_loc_y = p_loc_y + (y3 - p_loc_y) / smooth  # 当前鼠标所在的位置坐标y
            # (8)移动鼠标
            pyautogui.moveTo(c_loc_x, c_loc_y)  # 给出鼠标移动位置坐标
            print(detector.findDistance((thumb_tip_x, thumb_tip_y),
                                        (index_finger_pip_x, index_finger_pip_y),
                                        img)[0])
            if detector.findDistance((thumb_tip_x, thumb_tip_y),
                                     (index_finger_pip_x, index_finger_pip_y),
                                     img)[0] < 50:  # 4号关节和6号关节之间的距离小于20，就进行鼠标按压且移动
                pyautogui.mouseDown(x=c_loc_x, y=c_loc_y, button=pyautogui.LEFT)
                pyautogui.moveTo(x=c_loc_x, y=c_loc_y, duration=0.05)
            # 更新前一帧鼠标所在的位置坐标，将当前帧鼠标所在位置，变成下一帧的鼠标所在的前一阵位置
            p_loc_x, p_loc_y = c_loc_x, c_loc_y

        # 右手所有手指都弯曲则鼠标抬起
        if fingers[4] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
            pyautogui.mouseUp(button=pyautogui.LEFT)

    # 如果能检测到手部，并且是左手就可以进行鼠标滚轮的操作
    if hands and hands[0]['type'] == 'Left':
        lmList = hands[0]['lmList']  # hands是由N个字典组成的列表，其中每个字典包括每只手的关键点信息
        # 获取8、12、16、20这四个点位的位置信息
        index_finger_tip_x, index_finger_tip_y, index_finger_tip_z = lmList[8]
        middle_finger_tip_x, middle_finger_tip_y, middle_finger_tip_z = lmList[12]
        ring_finger_tip_x, ring_finger_tip_y, ring_finger_tip_z = lmList[16]  # 6号关节的位置
        pinky_tip_x, pinky_tip_y, pinky_tip_z = lmList[20]

        # 获取6、10、14、18这四个点位的位置信息
        index_finger_pip_x, index_finger_pip_y, index_finger_pip_z = lmList[6]
        middle_finger_pip_x, middle_finger_pip_y, middle_finger_pip_z = lmList[10]
        ring_finger_pip_x, ring_finger_pip_y, ring_finger_pip_z = lmList[14]  # 6号关节的位置
        pinky_pip_x, pinky_pip_y, pinky_pip_z = lmList[18]

        fingers = detector.fingersUp(hands[0])  # 检测hands中每根手指的朝向
        # 如果食指、中指、无名指、小拇指都是竖起，就认定鼠标向上滑动，每次滑动50个像素
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            cv2.circle(img=img, center=(index_finger_tip_x, index_finger_tip_y), radius=15, color=(0, 255, 0),
                       thickness=cv2.FILLED)
            cv2.circle(img=img, center=(middle_finger_tip_x, middle_finger_tip_y), radius=15, color=(0, 255, 0),
                       thickness=cv2.FILLED)
            cv2.circle(img=img, center=(ring_finger_tip_x, ring_finger_tip_y), radius=15, color=(0, 255, 0),
                       thickness=cv2.FILLED)
            cv2.circle(img=img, center=(pinky_tip_x, pinky_tip_y), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            pyautogui.scroll(+50)
        # 如果食指、中指、无名指、小拇指都是弯曲，就认定鼠标向下滑动，每次滑动50个像素
        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            cv2.circle(img=img, center=(index_finger_pip_x, index_finger_pip_y), radius=15, color=(0, 255, 0),
                       thickness=cv2.FILLED)
            cv2.circle(img=img, center=(middle_finger_pip_x, middle_finger_pip_y), radius=15, color=(0, 255, 0),
                       thickness=cv2.FILLED)
            cv2.circle(img=img, center=(ring_finger_pip_x, ring_finger_pip_y), radius=15, color=(0, 255, 0),
                       thickness=cv2.FILLED)
            cv2.circle(img=img, center=(pinky_pip_x, pinky_pip_y), radius=15, color=(0, 255, 0), thickness=cv2.FILLED)
            pyautogui.scroll(-50)

    # （10）显示图像
    # 查看FPS: 每秒FPS = 1秒钟能够处理多少帧 = 1秒钟 / 处理一帧所用的时间
    end_time = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
    # (end_time - start_time)表示处理一帧需要用多少秒,那么1 / (end_time - start_time)表示一秒可以处理多少帧，即fps
    fps = 1 / (end_time - start_time)

    # 在视频上显示fps信息
    # 图片，添加的文字(先转换成整数再变成字符串形式)，文本显示坐标，字体，字体大小，颜色，字体粗细
    cv2.putText(img, f"FPS: {str(int(fps))}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # (8)显示图像，输入窗口名及图像数据
    cv2.imshow('cap', img)
    if cv2.waitKey(20) & 0xff == 27:  # 每帧滞留20毫秒后消失，ESC键退出
        break

    if not cv2.getWindowProperty('cap', cv2.WND_PROP_VISIBLE):  # 窗口被点击右上角关闭按钮，则退出循环
        break

# 释放摄像头和视频资源
cap.release()
cv2.destroyAllWindows()
