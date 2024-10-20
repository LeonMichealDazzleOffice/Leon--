import pyautogui
import time
import threading
import os
import cv2
import numpy as np
import asyncio
from PIL import ImageGrab, Image
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor

# 初始化 PaddleOCR
ocr_model = PaddleOCR(use_angle_cls=False, lang='ch', show_log=False, use_gpu=False)  # 根据需要调整

# 定义区域坐标（left, upper, right, lower）
region1 = (1344, 273, 1398, 327)    # 左边数字位置
region2 = (1476, 270, 1550, 324)    # 右边数字位置
write_region = (1202, 510, 1626, 903)  # 绘制符号位置

task_count = 0
processed_questions = set()  # 存储已处理过的题目，以避免重复输入
lock = threading.Lock()

# 定义速度参数
move_duration = 0.05   # 鼠标移动的持续时间
draw_delay = 0.02      # 绘制点之间的延迟
recognition_delay = 0.15  # 识别的间隔时间，每秒识别 10 次

executor = ThreadPoolExecutor(max_workers=4)

def preprocess_image(image):
    """预处理图像以提高 OCR 识别效果。"""
    # 放大图像（例如放大2倍）
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    # 转换为灰度图像
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # 中值滤波去噪
    denoised = cv2.medianBlur(gray, 3)
    # 自适应阈值二值化
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    # 膨胀和腐蚀
    kernel = np.ones((2, 2), np.uint8)
    processed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 保存预处理后的图像以供检查
    cv2.imwrite("processed_image.png", processed_img)
    print("Image Preprocessing Complete. Saved as processed_image.png.")
    return processed_img

def recognize_number(image):
    """使用 PaddleOCR 识别图像中的数字。"""
    result = ocr_model.ocr(image, cls=False)
    print("OCR Result:", result)  # 调试信息
    if result and result != [None]:
        try:
            # 使用递归函数来展平多层嵌套列表
            def flatten(lst):
                for item in lst:
                    if isinstance(item, list):
                        yield from flatten(item)
                    else:
                        yield item

            # 展平 OCR 结果
            flat_result = list(flatten(result))
            print("Flattened OCR Result:", flat_result)  # 调试信息

            text = ''
            for item in flat_result:
                if isinstance(item, tuple) and len(item) == 2:
                    txt, conf = item
                    if isinstance(txt, str) and conf > 0.5:  # 检查置信度
                        text += txt
                else:
                    print("Skipping non-text item in flattened result:", item)

            print("Raw Text:", text)  # 调试信息
            filtered_text = ''.join(filter(str.isdigit, text))
            print("Filtered Text:", filtered_text)  # 调试信息

            if filtered_text:
                return int(filtered_text)
            else:
                print("Filtered text is empty.")
                return None
        except Exception as e:
            print(f"Error processing OCR result: {e}")
            return None
    else:
        print("No valid text found in OCR result.")
        return None

def draw_symbol(symbol):
    """在指定区域绘制符号，包含适当的延迟。"""
    x, y = write_region[0] + 150, write_region[1] + 150
    if symbol == '>':
        points = [(x, y), (x + 50, y + 50), (x, y + 100)]
    elif symbol == '<':
        points = [(x + 50, y), (x - 10, y + 50), (x + 50, y + 100), (x - 10, y + 75)]
    else:
        return

    # 绘制符号，包含延迟
    pyautogui.moveTo(points[0], duration=move_duration)
    pyautogui.mouseDown()
    for point in points[1:]:
        pyautogui.moveTo(point, duration=move_duration)
        time.sleep(draw_delay)
    pyautogui.mouseUp()
    time.sleep(0.2)

def update_task_count():
    """更新任务计数并每处理 10 次写入文件。"""
    global task_count
    with lock:
        task_count += 1
        if task_count % 10 == 0:
            file_path = 'tmcll.txt'
            with open(file_path, 'w') as f:
                f.write(str(task_count))
            print(f"累计处理响应：{task_count}")

async def process_numbers():
    """主循环，持续监控屏幕并处理数字比较。"""
    while True:
        try:
            # 截取屏幕区域
            screenshot1 = ImageGrab.grab(bbox=region1)
            screenshot2 = ImageGrab.grab(bbox=region2)

            # 保存截图以供检查
            screenshot1.save("screenshot1.png")
            screenshot2.save("screenshot2.png")
            print("Screenshots saved as screenshot1.png and screenshot2.png.")

            # 预处理图像
            processed_img1 = preprocess_image(screenshot1)
            processed_img2 = preprocess_image(screenshot2)

            # 异步处理 OCR 识别
            loop = asyncio.get_event_loop()
            future1 = loop.run_in_executor(executor, recognize_number, processed_img1)
            future2 = loop.run_in_executor(executor, recognize_number, processed_img2)

            number1, number2 = await asyncio.gather(future1, future2)
            print(f"Recognized Numbers: {number1}, {number2}")  # 调试信息

            if number1 is not None and number2 is not None:
                if number1 > number2:
                    symbol = '>'
                    print(f"比较数字：{number1} 和 {number2}，结果：大于")
                elif number1 < number2:
                    symbol = '<'
                    print(f"比较数字：{number1} 和 {number2}，结果：小于")
                else:
                    symbol = None
                    print(f"比较数字：{number1} 和 {number2}，结果：相等，无法绘制符号。")

                if symbol and (number1, number2) not in processed_questions:
                    threading.Thread(target=draw_symbol, args=(symbol,)).start()
                    update_task_count()
                    processed_questions.add((number1, number2))
                elif symbol and (number1, number2) in processed_questions:
                    time.sleep(0.4)
                    threading.Thread(target=draw_symbol, args=(symbol,)).start()
            else:
                print("无法正确识别数字。")

        except Exception as e:
            print(f"发生错误：{e}")

        # 异步休眠，控制识别频率
        await asyncio.sleep(recognition_delay)

if __name__ == "__main__":
    # 初始化任务计数
    if os.path.exists('tmcll.txt'):
        with open('tmcll.txt', 'r') as f:
            task_count = int(f.read())

    # 设置 pyautogui 的参数
    pyautogui.PAUSE = 0  # 取消 pyautogui 的默认暂停

    # 异步运行主函数
    asyncio.run(process_numbers())
 
