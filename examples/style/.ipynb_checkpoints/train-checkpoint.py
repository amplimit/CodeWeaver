# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import NeuralStyleTransferModel
import utils
import cv2
import argparse


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img_path', type=str, default='./images/1.jpg', help='原图路径')
    parser.add_argument('--style_img_path', type=str, default='./images/style.jpg', help='风格图片路径')
    parser.add_argument('--output_path', type=str, default='./output/1', help='生成图片保存路径')
    parser.add_argument('--epochs', type=int, default=20, help='total training epochs')
    parser.add_argument('--step_per_epoch', type=int, default=100, help='每个epoch训练次数')
    parser.add_argument('--learning_rate', type=int, default=0.01, help='学习率')
    parser.add_argument('--content_loss_factor', type=int, default=1, help='内容loss总加权系数')
    parser.add_argument('--style_loss_factor', type=int, default=100, help='风格loss总加权系数')
    parser.add_argument('--img_size', type=int, default=0, help='图片尺寸,0代表不设置使用默认尺寸(450*300),输入1代表使用图片尺寸,其他输入代表使用自定义尺寸')
    parser.add_argument('--img_width', type=int, default=450, help='自定义图片宽度')
    parser.add_argument('--img_height', type=int, default=300, help='自定义图片高度')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

opt = parse_opt()
print(opt)
CONTENT_LOSS_FACTOR = opt.content_loss_factor
STYLE_LOSS_FACTOR = opt.style_loss_factor
CONTENT_IMAGE_PATH = opt.content_img_path
STYLE_IMAGE_PATH = opt.style_img_path
OUTPUT_DIR = opt.output_path
EPOCHS = opt.epochs
LEARNING_RATE = opt.learning_rate
STEPS_PER_EPOCH = opt.step_per_epoch
# # 内容特征层及loss加权系数
# CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
# # 风格特征层及loss加权系数
# STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2,
#                 'block5_conv1': 0.2}


if opt.img_size==0:
    IMG_WIDTH = 450
    IMG_HEIGHT = 300
elif opt.img_size==1:
    #读取图片
    img = cv2.imread(CONTENT_IMAGE_PATH)
    IMG_WIDTH = img.shape[1]
    IMG_HEIGHT = img.shape[0]
else:
    IMG_WIDTH = opt.img_width
    IMG_HEIGHT = opt.img_height
print("IMG_WEIGHT:",IMG_WIDTH)
print("IMG_HEIGHT:",IMG_HEIGHT)



# 创建模型
model = NeuralStyleTransferModel()

# 加载内容图片
content_image = utils.load_images(CONTENT_IMAGE_PATH,IMG_WIDTH,IMG_HEIGHT)
# 风格图片
style_image = utils.load_images(STYLE_IMAGE_PATH,IMG_WIDTH,IMG_HEIGHT)

# 计算出目标内容图片的内容特征备用
target_content_features = model([content_image, ])['content']
# 计算目标风格图片的风格特征
target_style_features = model([style_image, ])['style']

# M = settings.WIDTH * settings.HEIGHT
M = IMG_WIDTH * IMG_HEIGHT
N = 3


def _compute_content_loss(noise_features, target_features):
    """
    计算指定层上两个特征之间的内容loss
    :param noise_features: 噪声图片在指定层的特征
    :param target_features: 内容图片在指定层的特征
    """
    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    # 计算系数
    x = 2. * M * N
    content_loss = content_loss / x
    return content_loss


def compute_content_loss(noise_content_features):
    """
    计算并当前图片的内容loss
    :param noise_content_features: 噪声图片的内容特征
    """
    # 初始化内容损失
    content_losses = []
    # 加权计算内容损失
    for (noise_feature, factor), (target_feature, _) in zip(noise_content_features, target_content_features):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


def gram_matrix(feature):
    """
    计算给定特征的格拉姆矩阵
    """
    # 先交换维度，把channel维度提到最前面
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape，压缩成2d
    x = tf.reshape(x, (x.shape[0], -1))
    # 计算x和x的逆的乘积
    return x @ tf.transpose(x)


def _compute_style_loss(noise_feature, target_feature):
    """
    计算指定层上两个特征之间的风格loss
    :param noise_feature: 噪声图片在指定层的特征
    :param target_feature: 风格图片在指定层的特征
    """
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    # 计算系数
    x = 4. * (M ** 2) * (N ** 2)
    return style_loss / x


def compute_style_loss(noise_style_features):
    """
    计算并返回图片的风格loss
    :param noise_style_features: 噪声图片的风格特征
    """
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(noise_style_features, target_style_features):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)


def total_loss(noise_features):
    """
    计算总损失
    :param noise_features: 噪声图片特征数据
    """
    content_loss = compute_content_loss(noise_features['content'])
    style_loss = compute_style_loss(noise_features['style'])
    return content_loss * CONTENT_LOSS_FACTOR + style_loss * STYLE_LOSS_FACTOR


# 使用Adma优化器
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# 基于内容图片随机生成一张噪声图片
# noise_image = tf.Variable((content_image + np.random.uniform(-0.2, 0.2, (1, settings.HEIGHT, settings.WIDTH, 3))) / 2)
noise_image = tf.Variable((content_image + np.random.uniform(-0.2, 0.2, (1, IMG_HEIGHT, IMG_WIDTH, 3))) / 2)

# 使用tf.function加速训练
@tf.function
def train_one_step():
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs)
    # 求梯度
    grad = tape.gradient(loss, noise_image)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients([(grad, noise_image)])
    return loss


def main():
    # 创建保存生成图片的文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # 共训练EPOCHS个epochs
    for epoch in range(EPOCHS):
        # 使用tqdm提示训练进度
        with tqdm(total=STEPS_PER_EPOCH, desc='Epoch {}/{}'.format(epoch + 1, EPOCHS)) as pbar:
            # 每个epoch训练STEPS_PER_EPOCH次
            for step in range(STEPS_PER_EPOCH):
                _loss = train_one_step()
                pbar.set_postfix({'loss': '%.4f' % float(_loss)})
                pbar.update(1)
            # 每个epoch保存一次图片
            print("{}\n{}\n{}".format(OUTPUT_DIR, CONTENT_IMAGE_PATH.split('.')[-2].split('/')[-1], epoch + 1))
            utils.save_image(noise_image, '{}_{}epoch.jpg'.format(CONTENT_IMAGE_PATH.split('.')[-2].split('/')[-1], epoch + 1))
            
if __name__ == '__main__':
    main()