# 数据反馈模块(主要完成绘图及文件保存功能)

import os
import shutil
import time
import numpy
from matplotlib import pyplot as plt

from format import DataPreprocessing
from optimize import InertiaSolver


class InertiaFeedback(DataPreprocessing, InertiaSolver):
    
    def __init__(self, path_ref, path_tar, path_res='./screens_res'):
        super(InertiaFeedback, self).__init__(path_ref, path_tar)
        self.path_res = path_res
        
    def predict_rgb(self, tar_index_list=None, ref_index_list=None, band_index_list=None, formatted=True, graphic=True):
        '''
        单屏: (Gray, weighted_distance) + label(mean)
        单屏: (Gray, current_distance) + label(mean)
        多屏: (screen_index, mean) + label(mean)
        band_index_list: 绘图band
        formatted: 是否将数据写入文本
        graphic: 是否进行绘图操作
        '''
        self.tar_index_list = range(self.number_tar) if tar_index_list is None else tar_index_list
        self.ref_index_list = range(self.number_ref) if ref_index_list is None else ref_index_list
        self.__formatted = formatted
        self.__graphic = graphic
        
        if self.__formatted or self.__graphic:
            self.__build_path_res()
        
        mean_std_screens = list()
        for tar_index in self.tar_index_list:
            self.get_D_and_A(tar_index, self.ref_index_list)
            
            GrayLmaxdict = self.tar_list[tar_index].get_sorted_GrayLmaxdict()
            bands_num = self.tar_list[tar_index].shape[0]
            
            bands_seq = list()
            for band_index, GrayLmax_band in GrayLmaxdict.items():
                bands_seq.append((band_index, GrayLmax_band[0][1]))
            bands_seq.sort(key=lambda item: item[1], reverse=True)
            
            # rgb_pred_curr_real = self.__solve(bands_seq, GrayLmaxdict)                             # 按照既定顺序求解 ###########################
            rgb_pred_curr_real = self.__solve_2(bands_seq, GrayLmaxdict)                           # 按照既定顺序求解 ###########################
            if band_index_list is None:
                band_index_list = range(bands_num)
            mean_std = self.__extract_and_display1(tar_index, band_index_list, GrayLmaxdict, rgb_pred_curr_real)
            mean_std_screens.append(mean_std)
        
        self.__extract_and_display2(mean_std_screens)
        
    def show_summary(self):
        if not os.path.exists('./summary.log'):
            raise Exception('>>> Missing the file named with "summary.log"! <<<')
        self.__show_summary()
        
    def __show_summary(self):
        screens_tar, screens_ref, pred_mean, curr_mean = list(), list(), list(), list()
        with open('./summary.log', 'rt') as f:
            for line in f:
                screens_tar.append(int(line[39:43]))
                screens_ref.append(int(line[58:62]))
                pred_mean.append(float(line[75:82]))
                curr_mean.append(float(line[95:102]))
                
        fig = plt.figure(figsize=(8, 6))
        ax1 = plt.subplot()
        ax1.plot(screens_ref, pred_mean, label='$pred\_mean$')
        ax1.plot(screens_ref, curr_mean, label='$curr\_mean$')
        ax1.set(xlabel='$ref\_screens\_count$', ylabel='$mean\_distance$')
        ax1.legend()
        fig.savefig('./summary.png', dpi=300)
        plt.close()
        
    def __build_path_res(self):
        if os.path.exists(self.path_res):
            shutil.rmtree(self.path_res)
            time.sleep(0.3)
        os.mkdir(self.path_res)
            
    # def __solve(self, bands_seq, GrayLmaxdict):
        # rgb_pred_curr_real = dict()                                     # {GrayLmax: (pred, curr, real)}
        
        # W0 = numpy.array([1 / self.curr_ref_num] * self.curr_ref_num).reshape(-1, 1)
        # for band_index, _ in enumerate(bands_seq):
            # GrayLmax_band = GrayLmaxdict[band_index]
            
            # W = W0
            # for idx, GrayLmax in enumerate(GrayLmax_band):
                # D_curr = self.D_and_A_init[GrayLmax][0]                 # 当前初值
            
                # A_stop = self.D_and_A_stop[GrayLmax][1]
                # D_pred = numpy.matmul(A_stop, W).round().astype(int)    # 预测初值
                
                # D_stop = self.D_and_A_stop[GrayLmax][0]                 # 真实值
                # rgb_pred_curr_real[GrayLmax] = (D_pred, D_curr, D_stop)
                # W = self.solve2norm(A_stop, D_stop)
                
                # if idx == 0:
                    # W0 = W
        
        # return rgb_pred_curr_real

    
    def __solve_2(self, bands_seq, GrayLmaxdict):
        self.C_dict = dict()              # {(idx0, idx): C, ...}
        rgb_pred_curr_real = dict()
        
        W0 = numpy.array([1 / self.curr_ref_num] * self.curr_ref_num).reshape(-1, 1)
        for idx0, (band_index, _) in enumerate(bands_seq):
            GrayLmax_band = GrayLmaxdict[band_index]
            
            W = W0
            for idx, GrayLmax in enumerate(GrayLmax_band):
                D_curr = self.D_and_A_init[GrayLmax][0]
                # 初值预测
                A_stop = self.D_and_A_stop[GrayLmax][1]
                D_pred = numpy.matmul(A_stop, W).round().astype(int)
                # 传统优化
                D_stop = self.D_and_A_stop[GrayLmax][0]
                rgb_pred_curr_real[GrayLmax] = (D_pred, D_curr, D_stop)
                # 权重优化
                C = self.__get_C(idx0, idx, GrayLmax_band)    # 计算正向二级权重
                W = self.solve2norm_2(A_stop, D_stop, C)
                
                ##########################################
                if idx == 0:
                    A_stop_0 = A_stop
                    D_stop_0 = D_stop
                if idx == 1:
                    C_1 = C
                ##########################################
                self.C_dict[(idx0, idx)] = C
            else:
                W0 = self.solve2norm_2(A_stop_0, D_stop_0, C_1)
        return rgb_pred_curr_real
    
    def __get_C(self, idx0, idx, GrayLmax_band):
        '''
        此处主要计算正向二级权重C
        idx0: band优化先后序号
        idx: 绑点优化先后序号
        GrayLmax_band: band上的绑点列表
        rgb_pred_curr_real: 已优化的rgb数值信息
        '''
        GrayLmax_curr = GrayLmax_band[idx]
        D_stop_curr = self.D_and_A_stop[GrayLmax_curr][0]
        A_stop_curr = self.D_and_A_stop[GrayLmax_curr][1]
        if idx > 0:
            GrayLmax_last = GrayLmax_band[idx - 1]
            D_stop_last = self.D_and_A_stop[GrayLmax_last][0]
            A_stop_last = self.D_and_A_stop[GrayLmax_last][1]
        
        if idx0 == 0:
            if idx == 0:
                C = numpy.identity(A_stop_curr.shape[1])
            else:
                C = self.__calc_C(D_stop_curr, A_stop_curr, D_stop_last, A_stop_last)   # 以夹角度量近似程度
        else:
            if idx == 0:
                C = self.C_dict[(idx0 - 1, 1)]
            else:
                C = self.__calc_C(D_stop_curr, A_stop_curr, D_stop_last, A_stop_last)
        
        return C
        
        
    def __calc_C(self, D_stop_curr, A_stop_curr, D_stop_last, A_stop_last, gamma=0.5):
        D_delta = D_stop_curr - D_stop_last
        A_delta = A_stop_curr - A_stop_last
        theta = self.__calc_theta(D_delta, A_delta)
        
        C = numpy.diag(numpy.exp(-(theta)**2 / gamma**2))
        return C
        
        
    def __calc_theta(self, D_delta, A_delta):
        inner_product = (D_delta * A_delta).sum(axis=0)
        cos_theta = inner_product / numpy.linalg.norm(D_delta) / numpy.linalg.norm(A_delta, axis=0) * 0.99999
        theta = numpy.arccos(cos_theta)
        return theta
        
    def __extract_and_display1(self, tar_index, band_index_list, GrayLmaxdict, rgb_pred_curr_real):
        filename = os.path.splitext(os.path.basename(self.filenames_tar[tar_index]))[0]
        filepath = os.path.join(self.path_res, filename)
        
        if self.__formatted:
            f = open('{}.log'.format(filepath), 'wt')
            f.write('Gray    Lmax  pred_dist curr_dist\n')
        
        if self.__graphic:
            fig = plt.figure(figsize=(23, 18))
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
        
        Gray_band = None
        rgb_distance = list()                                           # [(pred_distance, curr_distance)]
        for band_index, GrayLmax_band in GrayLmaxdict.items():
            
            if band_index in band_index_list:
                rgb_distance_band = list()
                Gray_band = list()
                
                for GrayLmax in GrayLmax_band:
                    pred_distance = numpy.linalg.norm(rgb_pred_curr_real[GrayLmax][0] - rgb_pred_curr_real[GrayLmax][2])
                    curr_distance = numpy.linalg.norm(rgb_pred_curr_real[GrayLmax][1] - rgb_pred_curr_real[GrayLmax][2])
                    rgb_distance_band.append((pred_distance, curr_distance))
                    Gray_band.append(GrayLmax[0])
                    if self.__formatted:
                        f.write('{:4d} {:8.2f} {:7.3f}   {:7.3f}\n'.format(*GrayLmax, pred_distance, curr_distance))
                    
                mean = numpy.array(rgb_distance_band).mean(axis=0)
                if self.__graphic:
                    ax1.plot(Gray_band, list(item[1] for item in rgb_distance_band), label='$band{},\ (Lmax={:8.2f},\ mean={:7.3f})$'.format(band_index, GrayLmax_band[0][1], mean[1]))
                    ax2.plot(Gray_band, list(item[0] for item in rgb_distance_band), label='$band{},\ (Lmax={:8.2f},\ mean={:7.3f})$'.format(band_index, GrayLmax_band[0][1], mean[0]))
                
                rgb_distance.extend(rgb_distance_band)
        
        rgb_distance = numpy.array(rgb_distance)
        max_distance = rgb_distance.max()
        
        if self.__graphic:
            xlim = [-1, 256]
            ax1.plot(xlim, [0, 0], color='k', linestyle='--')
            ax2.plot(xlim, [0, 0], color='k', linestyle='--')
            
            ax1.set(xlabel='$Gray$', xticks=Gray_band, ylabel='$curr\_rgb\_distance$', xlim=xlim, ylim=[-0.05 * max_distance, 1.05 * max_distance])
            ax2.set(xlabel='$Gray$', xticks=Gray_band, ylabel='$pred\_rgb\_distance$', xlim=xlim, ylim=[-0.05 * max_distance, 1.05 * max_distance])
            
            ax1.legend()
            ax2.legend()
            fig.tight_layout(pad=1.5)
            fig.savefig('{}.png'.format(filepath), dpi=300)
            plt.close()
        
        means_screen, stds_screen = rgb_distance.mean(axis=0), rgb_distance.std(axis=0)
        if self.__formatted:
            f.write('\nmean = {}\n'.format(means_screen))
            f.write('std  = {}'.format(stds_screen))
            f.close()
            
        return means_screen, stds_screen
        
    def __extract_and_display2(self, mean_std_screens):
        filename = os.path.join(self.path_res, 'all_screens.png')
        
        mean_list = list()
        for mean_std in mean_std_screens:
            mean_list.append(mean_std[0])
        mean_list = numpy.array(mean_list)
        
        if self.__graphic:
            fig = plt.figure(figsize=(8, 6))
            ax1 = plt.subplot()
            ax1.plot(range(mean_list.shape[0]), mean_list[:, 0], label='$pred$')
            ax1.plot(range(mean_list.shape[0]), mean_list[:, 1], label='$curr$')
            ax1.set(xlabel='$screen\_idx$', xticks=range(mean_list.shape[0]), ylabel='$mean\ distance$', title='$mean\ distance\ on\ each\ screen$')
            ax1.legend()
            fig.savefig(filename, dpi=300)
            plt.close()
        
        mean = mean_list.mean(axis=0)
        with open('./summary.log', 'at') as f:
            f.write('{}, screens_tar ={:4d}, screens_ref ={:4d}, pred_mean ={:7.3f}, curr_mean ={:7.3f};\n'.format(time.ctime(), len(self.tar_index_list), len(self.ref_index_list), mean[0], mean[1]))
                    
                
                
        



class PureFeedback(object):
    
    def __init__(self, path):
        '''
        不定绘图任务
        '''
        pass
        
    def show_rgb_on_Gray(self, screen_index_list=None, band_index_list=None):
        '''
        (r/g/b, Gray)
        (r/g/b, Gray_slope)
        '''
        pass
            
        
        
        
        
if __name__ == '__main__':
    path_ref = './screens_ref'
    path_tar = './screens_tar'
    
    obj = InertiaFeedback(path_ref, path_tar)
    
    for ref_number in range(5, 201):
        obj.predict_rgb(range(20), range(ref_number), formatted=False, graphic=False)
        
    obj.show_summary()





