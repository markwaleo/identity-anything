import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import threading
import tempfile
import shutil
import json
import base64
import os
import time

import openai
import cv2
import numpy as np
import moviepy.editor as mpe
import pygame
from PIL import Image, ImageDraw, ImageFont
from aip import AipSpeech

# --------------------------
# 全局常量及初始化部分
# --------------------------

# 视频模板文件路径
video_src_path = "./video.mp4"

# 透明叠加图片(展示框)路径
album_path = "./album.png"

# 尝试读取透明叠加图片并转换为NumPy数组
try:
    pil_img = Image.open(album_path).convert("RGBA")  # 使用PIL打开并转换为RGBA模式
    album_img = np.array(pil_img)                    # 转为NumPy数组
except Exception as e:
    messagebox.showerror("错误", f"无法读取 '展示框透明.png' 文件：\n{e}")
    exit()  # 如果读取失败，弹窗提示并退出程序

# 叠加图在视频中的坐标控制 (x1, y1) - (x2, y2)
x1, y1 = 680, 220
x2, y2 = 1580, 1270
overlay_width = x2 - x1   # 叠加图宽度
overlay_height = y2 - y1  # 叠加图高度

# 控制不同图像叠加在视频中出现的时间点
binary_start = 1.2      # 二值化图像显示的起始秒
binary_end = 6.0        # 二值化图像显示的结束秒
original_start = 6.2    # 原始图像开始显示的秒

# 文字在视频中的显示位置
text_pos = (1800, 400)

is_uploaded=False  # 是否以上传图片
# 初始化pygame和其音频模块
pygame.init()
pygame.mixer.init()

# --------------------------
# 获取本地API Key
# --------------------------
def get_api_key(num):
    """
    从key.txt文件的指定行读取API Key。
    :param num: 行号（从0开始）
    :return: 对应行的字符串API Key
    """
    with open("key.txt", 'r') as file:
        return file.readlines()[num].strip()

# 从key.txt中读取OpenAI和百度相关Key
openai.api_key = get_api_key(0)
openai.api_base = get_api_key(1)
BAIDU_APP_ID = get_api_key(2)
BAIDU_API_KEY = get_api_key(3)
BAIDU_SECRET_KEY = get_api_key(4)

# 初始化百度语音合成客户端
speech_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)

# --------------------------
# 全局变量
# --------------------------
current_video_file = None  # 当前合成后的视频临时文件路径
composition_data = {}      # 存储合成过程中需要的数据信息

# --------------------------
# 函数定义部分
# --------------------------

def encode_image(image_path):
    """
    将指定图片转换为Base64编码字符串。
    :param image_path: 图片文件路径
    :return: Base64编码后的字符串，若失败则返回None
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except:
        messagebox.showerror("错误", "未找到图片文件！")
        return None

def parse_gpt_response(response_text):
    """
    解析GPT-4o返回的JSON数据，提取主体(object)、矩形坐标(Left_up和right_down)。
    :param response_text: GPT返回的JSON格式字符串
    :return: (object_name, left_up, right_down)
             如果解析失败则返回 (None, None, None)
    """
    try:
        d = json.loads(response_text)  # 尝试将字符串转为JSON对象
        global is_uploaded
        is_uploaded=True
        return d.get("object",""), tuple(d.get("Left_up",[0,0])), tuple(d.get("right_down",[0,0]))
    except Exception as e:
        messagebox.showerror("解析错误", f"解析 GPT-4o 返回数据失败！\n错误详情：{e}")
        return None, None, None

def binarize_image(image_path):
    """
    读取图像并进行二值化处理(OTSU)。
    :param image_path: 图像路径
    :return: 二值化后的图像（灰度形式的NumPy数组），若读取失败返回None
    """
    # 按原格式读取图像（可能带透明通道）
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        messagebox.showerror("错误", "无法读取图片，请检查路径或图像格式！")
        return

    # 如果是 BGRA(四通道)，先转到 BGR，再转到 GRAY
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 然后再执行 Otsu 二值化
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_img

def synthesize_speech(text, output_file):
    """
    调用百度语音合成，将文本合成为音频文件并保存。
    :param text: 待合成的文字
    :param output_file: 输出MP3文件路径
    :return: True合成成功，False合成失败
    """
    # 传入文本到百度客户端进行语音合成
    result = speech_client.synthesis(text, 'zh', 1, {'vol': 5})
    if not isinstance(result, dict):
        # 如果结果不是dict，说明合成成功，则写入文件
        with open(output_file, 'wb') as f:
            f.write(result)
        return True
    else:
        # 如果返回的是dict，说明出现错误
        messagebox.showerror("语音合成错误", f"语音合成错误：{result}")
        return False

def create_text_image(text, size=(1280, 720), max_fontsize=500, min_fontsize=50,
                      text_color="yellow", bg_color=(255, 255, 255, 0),
                      font_path=r"C:\Windows\Fonts\SIMYOU.TTF"):
    """
    动态创建纯文字图片(PNG, RGBA)，文字居中显示，并根据文字长度自适应字号。
    :param text: 要绘制的文字
    :param size: 图片的宽高(像素)
    :param max_fontsize: 最大字号
    :param min_fontsize: 最小字号
    :param text_color: 文字颜色
    :param bg_color: 背景颜色(默认为透明背景)
    :param font_path: 字体路径
    :return: PIL.Image对象
    """
    # 创建一张指定大小、指定背景色的RGBA图像
    img = Image.new("RGBA", size, bg_color)
    draw = ImageDraw.Draw(img)

    # 初始从最大字号开始尝试
    fontsize = max_fontsize
    font = ImageFont.truetype(font_path, fontsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 循环减小字号，直到文字能放入图片或低于最小字号
    while (w > size[0] or h > size[1]) and fontsize > min_fontsize:
        fontsize -= 10
        font = ImageFont.truetype(font_path, fontsize)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 计算居中绘制时的左上角坐标
    x = (size[0] - w) // 2
    y = (size[1] - h) // 2

    # 在图像上绘制文本
    draw.text((x, y), text, font=font, fill=text_color)
    return img

def play_video_with_pygame(video_file):
    """
    用pygame的preview功能播放指定的视频文件。
    :param video_file: 视频文件路径
    """
    pygame.display.quit()
    pygame.display.init()

    # 读取视频并根据屏幕尺寸判断是否需要缩放
    clip = mpe.VideoFileClip(video_file)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min(sw / clip.w, sh / clip.h)
    if scale < 1:
        clip = clip.resize(scale)

    clip.preview()  # 调用moviepy的预览接口
    clip.close()

def preview_materials(video_src_path, binary_overlay_rgb, original_overlay_rgb,
                      album_img, text_img, x1y1, text_pos,
                      binary_start, binary_end, original_start,
                      tts_file=None):
    """
    通过pygame的方式实时预览视频+叠加效果(二值化、原图、透明PNG、文字、混音)。
    :param video_src_path: 视频模板路径
    :param binary_overlay_rgb: 二值化图像的RGB数组
    :param original_overlay_rgb: 原图的RGB数组
    :param album_img: 透明背景(叠加框)的NumPy数组
    :param text_img: 文字图片(PIL.Image)
    :param x1y1: (x1, y1)叠加图绘制到视频的左上坐标
    :param text_pos: 文字叠加在视频中的位置
    :param binary_start: 二值化图像显示起始秒
    :param binary_end: 二值化图像显示结束秒
    :param original_start: 原图开始显示秒
    :param tts_file: 语音合成文件路径(可选)
    """
    # 尝试读取视频
    try:
        video_clip = mpe.VideoFileClip(video_src_path)
    except Exception as e:
        messagebox.showerror("视频错误", f"无法打开视频文件：{e}")
        return

    # 计算缩放系数，让视频适配当前屏幕大小
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    sc = min(1, sw / video_clip.w, sh / video_clip.h)
    nw, nh = int(video_clip.w * sc), int(video_clip.h * sc)

    # 创建pygame窗口
    screen = pygame.display.set_mode((nw, nh))
    pygame.display.set_caption("预览播放")

    # 获取叠加图、文字图在缩放后的坐标位置
    ox1, oy1 = x1y1
    sx1, sy1 = int(ox1 * sc), int(oy1 * sc)
    stx, sty = int(text_pos[0] * sc), int(text_pos[1] * sc)

    # 小工具：将NumPy图像数据转成pygame可显示的Surface
    def to_surface(arr):
        return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))

    # 转换二值化图、原图、叠加框以及文字图为pygame可用的Surface
    bin_surf = to_surface(binary_overlay_rgb)
    org_surf = to_surface(original_overlay_rgb)
    alb_surf = (pygame.image.frombuffer(album_img.tobytes(), (album_img.shape[1], album_img.shape[0]), "RGBA")
                if album_img.ndim == 3 and album_img.shape[2] == 4 else to_surface(album_img))
    t_rgba = text_img.convert("RGBA")
    t_surf = pygame.image.frombuffer(t_rgba.tobytes(), t_rgba.size, "RGBA")

    # 缩放上述Surface以适应屏幕缩放
    bin_surf = pygame.transform.scale(bin_surf, (int(bin_surf.get_width() * sc), int(bin_surf.get_height() * sc)))
    org_surf = pygame.transform.scale(org_surf, (int(org_surf.get_width() * sc), int(org_surf.get_height() * sc)))
    alb_surf = pygame.transform.scale(alb_surf, (int(alb_surf.get_width() * sc), int(alb_surf.get_height() * sc)))
    t_surf = pygame.transform.scale(t_surf, (int(t_surf.get_width() * sc), int(t_surf.get_height() * sc)))

    temp_audio_file = None

    # 如果视频带音频，则提取出来并与TTS进行混音
    if video_clip.audio is not None:
        try:
            # 提取视频的原音轨到临时文件
            temp_bg = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            video_clip.audio.write_audiofile(temp_bg, fps=44100, logger=None)

            # 如果有TTS文件则混音，否则只用原音轨
            if tts_file and os.path.exists(tts_file):
                bg_audio = mpe.AudioFileClip(temp_bg)
                tts_audio = mpe.AudioFileClip(tts_file).set_start(9.2)  # TTS晚9.2秒开始
                final_audio = mpe.CompositeAudioClip([bg_audio, tts_audio])

                # 写入一个新的临时文件
                temp_audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                final_audio.write_audiofile(temp_audio_file, fps=44100, logger=None)

                # 关闭音频资源
                final_audio.close()
                bg_audio.close()
                tts_audio.close()
            else:
                # 只有原音轨
                temp_audio_file = temp_bg

            # 用pygame.mixer加载最终音频并播放
            pygame.mixer.music.load(temp_audio_file)
            pygame.mixer.music.play()

        except Exception as e:
            messagebox.showerror("音频错误", f"提取或合并音频失败：{e}")

    clock = pygame.time.Clock()
    start_time = time.time()
    running = True

    # 循环按时间播放视频并实时叠加二值化图、原图、透明PNG、文字等
    while running:
        t = time.time() - start_time
        if t > video_clip.duration:
            # 如果播放时间超出视频时长，则退出循环
            running = False
            break

        # 获取当前帧并转为Surface
        frame = video_clip.get_frame(t)
        fsurf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        fsurf = pygame.transform.scale(fsurf, (nw, nh))
        screen.blit(fsurf, (0, 0))

        # 在指定时间段内叠加二值化图
        if binary_start <= t < binary_end:
            screen.blit(bin_surf, (sx1, sy1))
            screen.blit(alb_surf, (0, 0))

        # original_start之后叠加原图和文字
        if t >= original_start:
            screen.blit(org_surf, (sx1, sy1))
            screen.blit(alb_surf, (0, 0))
            screen.blit(t_surf, (stx, sty))

        pygame.display.flip()

        # 处理pygame事件，如关闭窗口
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.mixer.music.stop()
                pygame.quit()
                return

        clock.tick(video_clip.fps)

    # 清理
    video_clip.close()
    if temp_audio_file:
        pygame.mixer.music.stop()
        try:
            os.remove(temp_audio_file)  # 删除临时音频文件
        except Exception:
            pass
    pygame.quit()

def compose_and_save_video():
    """
    将预览效果真正合成到视频并保存到用户指定路径。
    合成内容包含：视频、二值化图像、原图、透明PNG、文字以及可能的混音。
    """
    global is_uploaded

    if is_uploaded is False:
        messagebox.showinfo("提示","请先生成视频！")
        return

    # 让用户选择保存位置
    save_path = filedialog.asksaveasfilename(
        title="保存视频", defaultextension=".mp4", filetypes=[("MP4文件", "*.mp4")]
    )
    if not save_path:
        return

    text_display.insert(tk.END, "开始缓存视频到本地\n")
    text_display.see(tk.END)

    # 读取模板视频
    try:
        video_clip = mpe.VideoFileClip(video_src_path)
    except Exception as e:
        messagebox.showerror("视频错误", f"无法打开视频文件：{e}")
        return

    # 从composition_data中获取二值化图、原图、叠加坐标、时间点等信息
    b_rgb = composition_data["binary_overlay_rgb"]
    o_rgb = composition_data["original_overlay_rgb"]
    x_ = composition_data["x1"]
    y_ = composition_data["y1"]

    # 构造MoviePy的ImageClip来在指定时间段叠加
    binary_clip = (mpe.ImageClip(b_rgb)
                   .set_start(composition_data["binary_start"])
                   .set_duration(composition_data["binary_end"] - composition_data["binary_start"])
                   .set_pos((x_, y_)))

    original_clip = (mpe.ImageClip(o_rgb)
                     .set_start(composition_data["original_start"])
                     .set_duration(video_clip.duration - composition_data["original_start"])
                     .set_pos((x_, y_)))

    # 叠加透明PNG(展示框)
    album_clip1 = (mpe.ImageClip(album_img)
                   .set_start(composition_data["binary_start"])
                   .set_duration(composition_data["binary_end"] - composition_data["binary_start"])
                   .set_pos((0, 0)))
    album_clip2 = (mpe.ImageClip(album_img)
                   .set_start(composition_data["original_start"])
                   .set_duration(video_clip.duration - composition_data["original_start"])
                   .set_pos((0, 0)))

    # 在最终画面中叠加文字
    object_name = composition_data["object_name"]
    text_img = create_text_image(object_name)
    temp_text_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    text_img.save(temp_text_file)  # 先保存到临时PNG文件
    text_overlay = (mpe.ImageClip(temp_text_file)
                    .set_start(composition_data["original_start"])
                    .set_duration(video_clip.duration - composition_data["original_start"])
                    .set_pos(composition_data["text_pos"]))

    # 通过CompositeVideoClip合成所有画面
    composite_clip = mpe.CompositeVideoClip([
        video_clip,
        binary_clip,
        album_clip1,
        original_clip,
        album_clip2,
        text_overlay
    ])

    # 处理混音部分，如果有tts文件则叠加
    tts_path = composition_data.get("tts_path", None)
    final_audio = None
    if tts_path and os.path.exists(tts_path):
        try:
            synth_audio_clip = mpe.AudioFileClip(tts_path).set_start(9.2)
            if video_clip.audio is not None:
                final_audio = mpe.CompositeAudioClip([video_clip.audio, synth_audio_clip])
            else:
                final_audio = synth_audio_clip
        except Exception as e:
            messagebox.showerror("语音错误", f"无法加载合成语音：{e}")
            return
    else:
        final_audio = video_clip.audio

    # 设置合成后的视频音轨
    composite_clip.audio = final_audio

    # 输出到临时视频文件
    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    composite_clip.write_videofile(
        temp_video_file, audio=True, codec="libx264", audio_codec="aac",
        audio_bitrate="192k", fps=video_clip.fps, logger=None
    )

    # 关闭资源
    composite_clip.close()
    video_clip.close()

    # 更新current_video_file
    current_video_file = temp_video_file
    progress_bar["value"] = 100
    progress_bar.update()
    messagebox.showinfo("合成完成", "视频合成完成！")

    # 将临时文件复制到用户指定的保存路径
    try:
        shutil.copy(current_video_file, save_path)
        messagebox.showinfo("保存成功", f"视频已保存到: {save_path}")
    except Exception as e:
        messagebox.showerror("保存失败", f"保存视频失败：{e}")

def process_image_and_play():
    """
    点击“选择图片”按钮后执行的主流程：
      1. 让用户选择一张图片
      2. 将图片进行Base64编码并发送给GPT-4o分析得到主体与坐标
      3. 调用百度TTS合成音频
      4. 做二值化处理，并生成预览播放所需的数据(保存在composition_data)
      5. 开启一个线程来预览播放视频效果
    """
    global current_video_file, composition_data

    # 让用户选择图片
    image_path = filedialog.askopenfilename(
        title="选择图片", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if not image_path:
        return

    # 在文本框里提示当前状态
    text_display.insert(tk.END, "正在处理图片...\n")
    text_display.see(tk.END)
    progress_bar["value"] = 0
    progress_bar.update()

    # 将图片编码为Base64
    base64_image = encode_image(image_path)
    if not base64_image:
        return

    # 读取图片以获取其宽高
    img = Image.open(image_path)
    width, height = img.size

    # 准备发送给GPT的提示文本
    text_input = (
        f"请分析这张图片并用尽量少的字描述其中的主体内容，如果物体为主要内容则只讲主体无需背景，"
        f"如果没有主要物体则讲场景，并告诉我矩形坐标，图片大小为{width}x{height}。"
        f"回复格式: {{\"object\":\"羊\", \"Left_up\":[300,200], \"right_down\":[750,600]}}"
    )

    # 调用GPT-4o接口进行分析
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个可以分析图片和文字的 AI 助手。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
    except Exception as e:
        messagebox.showerror("OpenAI 错误", f"调用 GPT-4o 失败：{e}")
        return

    # 解析GPT返回内容
    gpt_response = response["choices"][0]["message"]["content"]
    text_display.insert(tk.END, "GPT-4o 分析结果:\n" + gpt_response + "\n")
    text_display.see(tk.END)
    object_name, ret_left_up, ret_right_down = parse_gpt_response(gpt_response)
    if not (ret_left_up and ret_right_down):
        return

    # 更新进度
    progress_bar["value"] = 20
    progress_bar.update()
    text_display.insert(tk.END, "开始合成语音\n")
    text_display.see(tk.END)

    # 百度TTS合成
    global temp_tts_file
    temp_tts_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    if not synthesize_speech(object_name, temp_tts_file):
        return

    progress_bar["value"] = 30
    progress_bar.update()
    text_display.insert(tk.END, "开始图像处理\n")
    text_display.see(tk.END)

    # 读取原图并调整尺寸到overlay区域
    orig_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if orig_img is None:
        messagebox.showerror("错误", "读取图片失败！")
        return
    global original_overlay_rgb
    original_overlay = cv2.resize(orig_img, (overlay_width, overlay_height))
    original_overlay_rgb = cv2.cvtColor(original_overlay, cv2.COLOR_BGR2RGB)

    # 进行二值化处理
    binary_img = binarize_image(image_path)
    if binary_img is None:
        return
    global binary_overlay_rgb
    binary_img_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    binary_overlay = cv2.resize(binary_img_bgr, (overlay_width, overlay_height))
    binary_overlay_rgb = cv2.cvtColor(binary_overlay, cv2.COLOR_BGR2RGB)

    progress_bar["value"] = 60
    progress_bar.update()

    # 检测模板视频是否能正常打开
    try:
        _ = mpe.VideoFileClip(video_src_path)
    except Exception as e:
        messagebox.showerror("视频错误", f"无法打开视频文件：{e}")
        return

    # 为预览和最终合成生成一个文字图
    global text_img_created
    text_img_created = create_text_image(object_name)

    progress_bar["value"] = 80
    progress_bar.update()
    text_display.insert(tk.END, "开始合成视频\n")
    text_display.see(tk.END)

    # 组装合成所需的数据到composition_data，供后续合成或预览使用
    composition_data = {
        "video_src_path": video_src_path,
        "binary_overlay_rgb": binary_overlay_rgb,
        "original_overlay_rgb": original_overlay_rgb,
        "album_img": album_img,
        "text_img": text_img_created,
        "x1": x1,
        "y1": y1,
        "text_pos": text_pos,
        "binary_start": binary_start,
        "binary_end": binary_end,
        "original_start": original_start,
        "object_name": object_name,
        "tts_path": temp_tts_file
    }

    pygame.display.quit()
    pygame.mixer.quit()
    pygame.display.init()
    pygame.mixer.init()
    # 启动线程来进行预览播放，以免阻塞UI
    threading.Thread(target=preview_materials, args=(
        video_src_path, binary_overlay_rgb, original_overlay_rgb,
        album_img, text_img_created, (x1, y1), text_pos,
        binary_start, binary_end, original_start,
        temp_tts_file
    )).start()

    progress_bar["value"] = 100
    progress_bar.update()

def replay_video():
    """
    点击“重复播放”按钮后执行：
      - 如果已经有合成后的current_video_file，则播放此视频；
      - 否则如果合成步骤已完成，则再次调用preview_materials进行预览；
      - 如果都没数据，则提示用户先选择图片生成视频。
    """
    if current_video_file:
        # 播放最终合成好的视频文件
        threading.Thread(target=play_video_with_pygame, args=(current_video_file,)).start()
    elif progress_bar["value"] == 100:
        # 如果进度到100说明预览已完成，但还没有最终视频
        pygame.display.quit()
        pygame.mixer.quit()
        pygame.display.init()
        pygame.mixer.init()
        threading.Thread(target=preview_materials, args=(
            video_src_path, binary_overlay_rgb, original_overlay_rgb,
            album_img, text_img_created, (x1, y1), text_pos,
            binary_start, binary_end, original_start,
            temp_tts_file
        )).start()
    else:
        messagebox.showinfo("提示", "请先选择图片生成视频！")

# --------------------------
# Tkinter 界面部分
# --------------------------
root = tk.Tk()
root.title("神奇宝贝万能识别")
root.state('zoomed')  # 全屏(最大化)窗口

# 文本展示区域
text_display = tk.Text(root, height=10, width=80)
text_display.pack(pady=10)

# 进度条
progress_bar = Progressbar(root, orient="horizontal", length=500, mode="determinate")
progress_bar.pack(pady=10)

# “选择图片”按钮
btn_select = tk.Button(root, text="选择图片", command=process_image_and_play, font=("Arial", 14))
btn_select.pack(pady=10)

# “保存视频”按钮
btn_save = tk.Button(root, text="保存视频", command=compose_and_save_video, font=("Arial", 14))
btn_save.pack(pady=10)

# “重复播放”按钮
btn_replay = tk.Button(root, text="重复播放", command=replay_video, font=("Arial", 14))
btn_replay.pack(pady=10)

# 启动主循环
root.mainloop()
