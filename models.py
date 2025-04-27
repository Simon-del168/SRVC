import tensorflow as tf


def srvc_base(lr, scale, F, block_h, block_w):
    est = lr  # 初始化输出为输入LR图像（用于后续逐层叠加）

     # 步骤1：空间到批量转换（分块处理，便于后续局部操作）
    patches = tf.space_to_batch_nd(lr, block_shape=[block_h, block_w], paddings=[[0, 0], [0, 0]])
    # - 将输入图像按[block_h, block_w]分块，填充为整除块大小（paddings设为0，假设输入尺寸已适配）
    # - 输出形状：[batch*block_h*block_w, height/block_h, width/block_w, channels]
    # - 作用：将空间分块转换为批量维度，便于并行处理局部区域

    # 步骤2：特征提取（第一层卷积）
    features = tf.layers.conv2d(patches, 256, 3, strides=(1, 1), padding='valid',  # 输出256通道，3x3卷积
                                data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu,
                                use_bias=True)
    # - 提取底层特征（边缘、纹理等），ReLU激活引入非线性

     # 步骤3：动态生成卷积核和偏置
    kernel = tf.layers.conv2d(features, 3 * 3 * 3 * F, 3, strides=(1, 1), padding='valid',
                              data_format='channels_last', dilation_rate=(1, 1), activation=None,
                              use_bias=True)
    bias = tf.layers.conv2d(features, F, 3, strides=(1, 1), padding='valid',
                            data_format='channels_last', dilation_rate=(1, 1), activation=None,
                            use_bias=True)
        # - 通过卷积生成动态滤波器参数（kernel和bias），而非固定权重
    # - 假设输入通道为3（RGB），则每个输出通道的3x3核需要3x3x3参数，共3x3x3xF参数

     # 步骤4：重塑核形状以适配矩阵乘法
    kernel = tf.reshape(kernel, [-1, 1, 1, 3 * 3 * 3, F])
    bias = tf.reshape(bias, [-1, 1, 1, F])

    # 步骤5：提取图像块
    patches = tf.image.extract_patches(patches, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                       padding='SAME')
    # - 从分块后的图像中提取3x3的局部补丁（sizes指定补丁大小为[1,3,3,1]，即空间3x3，通道1）
    # - 输出形状：[batch, h', w', 9, 3]（每个补丁展平为9个像素，3通道）
    
    patches = tf.expand_dims(patches, axis=3)    # 增加维度：[batch, h', w', 1, 9, 3]

    # 步骤6：矩阵乘法应用动态滤波器
    patches = tf.matmul(patches, kernel)
    patches = tf.squeeze(patches, axis=3) + bias
    patches = tf.nn.relu(patches)

     # 步骤7：批量到空间转换
    est = tf.batch_to_space_nd(patches, block_shape=[block_h, block_w], crops=[[0, 0], [0, 0]])

    # 步骤8：后续卷积层（特征融合与上采样准备）
    est = tf.layers.conv2d(est, 128, 5, strides=(1, 1), padding='same',
                           data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu,
                           use_bias=True)
    est = tf.layers.conv2d(est, 32, 3, strides=(1, 1), padding='same',
                           data_format='channels_last', dilation_rate=(1, 1), activation=tf.nn.relu,
                           use_bias=True)

     # 步骤9：上采样层（通过深度到空间转换实现放大）
    est = tf.layers.conv2d(est, 3 * scale * scale, 3, strides=(1, 1), padding='same',
                           data_format='channels_last', dilation_rate=(1, 1), activation=None,
                           use_bias=True)
    est = tf.nn.depth_to_space(est, scale, data_format='NHWC')
    indepth_est = est
    return indepth_est


def srvc(lr):
    scale = 4
    F = 32
    block_h = tf.shape(lr)[1] / 5
    block_w = tf.shape(lr)[2] / 5
    return srvc_base(lr, scale, F, block_h, block_w)
