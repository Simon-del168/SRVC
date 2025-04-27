import numpy as np
import tensorflow as tf

//计算两个 RGB 图像之间的结构相似性指数（SSIM）和峰值信噪比（PSNR），其中 PSNR 分别计算 Y 通道（亮度通道）和 RGB 整体的结果
def get_metrics(im1, im2):
    # Assumes im1 and im2 are in RGB space with float values in range [0, 1]
    im1 = tf.clip_by_value(im1, 0, 1)
    im2 = tf.clip_by_value(im2, 0, 1)
    im1_y = tf.image.rgb_to_yuv(im1)[...,0]
    im2_y = tf.image.rgb_to_yuv(im2)[...,0]
    psnr_y = tf.reduce_mean(tf.image.psnr(im1_y, im2_y, max_val=1.))
    psnr_rgb = tf.reduce_mean(tf.image.psnr(im1, im2, max_val=1.))
    # 计算SSIM（Structural Similarity Index，结构相似性）
    # SSIM从亮度、对比度、结构三方面评估图像相似性，更符合人眼感知
    ssim = tf.reduce_mean(tf.image.ssim(im1, im2, max_val=1.))
    metrics = {'ssim': ssim, 'psnr_y': psnr_y, 'psnr_rgb': psnr_rgb}
    return metrics
//模型参数保存与恢复 SaveHelper
//该类用于管理 TensorFlow 模型的变量保存（到文件）和恢复（从文件加载），
//支持变量名映射（通过map_fun），适用于模型微调或多任务训练时的参数迁移。
class SaveHelper:
    def __init__(self, graph, map_fun):
        with graph.as_default():
            #graph：TensorFlow 计算图对象（模型定义所在的图）。
            #map_fun：变量名映射函数（例如，将模型 A 的变量名conv1/weights映射为模型 B 的new_conv1/weights），
            #用于解决不同模型变量名不匹配的问题。
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.vars_dict = {map_fun(v.name): v for v in variables}
            self.vars_pl = {
                map_fun(v.name): tf.placeholder(dtype=v.dtype, shape=v.shape, name='pl_%s' % (v.name.strip(':0'))) for v
                in variables}
            self.load_ops = {k: tf.assign(self.vars_dict[k], self.vars_pl[k], use_locking=True) for k in self.vars_dict}

    //将模型变量的值保存到字典或.npy文件，支持通过map_fun过滤或重命名变量。
    def save_vars(self, sess, vars_list, map_fun, save_dir=None):
        vars_vals = sess.run(vars_list)
        save_dict = {}
        for i in range(len(vars_list)):
            if map_fun(vars_list[i].name):
                save_dict[map_fun(vars_list[i].name)] = vars_vals[i]
        if save_dir:
            np.save(save_dir, save_dict)
        return save_dict

    //从.npy文件或字典加载变量值，并写入模型的对应变量中，支持通过map_fun过滤不匹配的变量。
    def restore_vars(self, sess, load_dir, map_fun):
        print('Trying to restore checkpoint')
        if isinstance(load_dir, str):
            vars_list = np.load(load_dir, allow_pickle=True).item()
        elif isinstance(load_dir, dict):
            vars_list = load_dir
        else:
            exit(1)
        restore_ops = [self.load_ops[var_name] for var_name in vars_list if not map_fun(var_name) is None]
        feed_dict = {self.vars_pl[var_name]: vars_list[var_name] for var_name in vars_list if
                     not map_fun(var_name) is None}
        sess.run(restore_ops, feed_dict=feed_dict)
        print('Restored successfully')
        return
