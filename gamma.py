# GammaTuning

import os
import shutil
import time

import numpy
from matplotlib import pyplot as plt

from parse.gamma_parse import DataPre
from optimize.gamma_inertia import InertiaSolver


class GTInitial(DataPre, InertiaSolver):
    
    def __init__(self, tar_path, ref_path, res_path='./screens_res'):
        super(GTInitial, self).__init__('LinkJCExcel')
        
        self.load_all_files(tar_path, ref_path)
        self.res_path = res_path
        
    def check_files(self, shape_save=None, printable=False):
        self.filter_by_shape(shape_save)
        
        tar_optional = self.files[self.tar_path]
        ref_optional = self.files[self.ref_path]
        
        if printable:
            print('optional tar_files: {}; optional ref_files: {}\n'.format(len(tar_optional), len(ref_optional)))
            for filename in tar_optional:
                print('tar: {}'.format(filename))
            print('')
            for filename in ref_optional:
                print('ref: {}'.format(filename))
        
        return len(tar_optional), len(ref_optional)
        
    def predict_rgb(self, omega=0, tar_index_list=None, ref_index_list=None, log_saved=True, fig_saved=True):
        self.__omega = omega
        self.tar_index_list = range(len(self.files[self.tar_path])) if tar_index_list is None else list(tar_index_list)
        self.ref_index_list = range(len(self.files[self.ref_path])) if ref_index_list is None else list(ref_index_list)
        
        self.__log_saved = log_saved
        self.__graph_saved = fig_saved
        
        if log_saved or fig_saved:
            self.__build_res_path()
        
        mean_std_screens = list()
        self.tar2rgb_pcr = dict()
        
        for tar_index in self.tar_index_list:
            self.get_D_and_A(tar_index, self.ref_index_list)
            
            tar_attrs = self.instances[self.tar_path][tar_index].get_attrs()
            band2Lmax = tar_attrs['band2Lmax']
            band2Lmax_new = sorted(list(band2Lmax.items()), key=lambda item: item[1], reverse=True)
            band2GrayLmax = tar_attrs['band2GrayLmax']
            
            rgb_pred_curr_real = self.__solve(band2Lmax_new, band2GrayLmax)
            self.tar2rgb_pcr[tar_index] = rgb_pred_curr_real
            
            mean_std = self.__extract_and_display1(tar_index, band2GrayLmax, rgb_pred_curr_real)
            mean_std_screens.append(mean_std)
        
        self.__extract_and_display2(mean_std_screens)    
            
    def show_summary(self):
        if not os.path.exists('./inertia_sum.log'):
            raise Exception('>>> Missing the file named with "summary.log"! <<<')
        
        self.__show_summary()
        
    def __show_summary(self):
        screens_tar, screens_ref, pred_mean, curr_mean = list(), list(), list(), list()
        with open('./inertia_sum.log', 'rt') as f:
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
        fig.savefig('./inertia_sum.png', dpi=300)
        plt.close()
        
    def __build_res_path(self):
        if os.path.exists(self.res_path):
            shutil.rmtree(self.res_path)
            time.sleep(0.3)
        os.mkdir(self.res_path)
        
    def __solve(self, band2Lmax_new, band2GrayLmax):
        rgb_pred_curr_real = dict()
        
        W0 = numpy.ones(shape=(len(self.ref_index_list), 1)) / len(self.ref_index_list)
        for band, _ in band2Lmax_new:
            GrayLmax_band = reversed(band2GrayLmax[band])

            W = W0
            for idx, GrayLmax in enumerate(GrayLmax_band):
                D_curr = self.init_D_and_A[GrayLmax][0]
                # 初值预测
                A_stop = self.stop_D_and_A[GrayLmax][1]
                D_pred = numpy.matmul(A_stop, W).round().astype(int)
                # 传统优化
                D_stop = self.stop_D_and_A[GrayLmax][0]
                rgb_pred_curr_real[GrayLmax] = numpy.hstack((D_pred, D_curr, D_stop))
                # 权重更新
                W = self.solve2norm_new(A_stop, D_stop, W, omega=self.__omega)             #########################
                
                if idx == 0:
                    W0 = W
        
        return rgb_pred_curr_real
        
    def __extract_and_display1(self, tar_index, band2GrayLmax, rgb_pred_curr_real):
        filename = os.path.splitext(self.files[self.tar_path][tar_index])[0]
        filepath = os.path.join(self.res_path, filename)
        
        if self.__log_saved:
            f = open('{}.log'.format(filepath), 'wt')
            f.write('Gray    Lmax  pred_dist curr_dist\n')
            
        if self.__graph_saved:
            fig = plt.figure(figsize=(23, 15))
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
        
        Gray_band = None
        rgb_distance = list()
        for band, GrayLmax_band in band2GrayLmax.items():
            rgb_distance_band = list()
            Gray_band = list(Gray for Gray, Lmax in GrayLmax_band)
            
            for GrayLmax in GrayLmax_band:
                pcr_distance = numpy.linalg.norm(rgb_pred_curr_real[GrayLmax] - rgb_pred_curr_real[GrayLmax][:, -1].reshape(-1, 1), axis=0)
                rgb_distance_band.append(pcr_distance[0:2])
                
                if self.__log_saved:
                    f.write('{:4.0f} {:8.2f} {:7.3f}   {:7.3f}\n'.format(*GrayLmax, pcr_distance[0], pcr_distance[1]))
                
            mean_band = numpy.array(rgb_distance_band).mean(axis=0)
            
            if self.__graph_saved:
                ax1.plot(Gray_band, list(item[0] for item in rgb_distance_band), label='$band{},\ (Lmax={:8.2f},\ mean={:7.3f})$'.format(int(band), GrayLmax_band[0][1], mean_band[0]))
                ax2.plot(Gray_band, list(item[1] for item in rgb_distance_band), label='$band{},\ (Lmax={:8.2f},\ mean={:7.3f})$'.format(int(band), GrayLmax_band[0][1], mean_band[1]))
            
            rgb_distance.extend(rgb_distance_band)
            
        rgb_distance = numpy.array(rgb_distance)
        max_distance = rgb_distance.max()
        
        if self.__graph_saved:
            xlim = [-1, 256]
            ax1.plot(xlim, [0, 0], color='k', linestyle='--')
            ax2.plot(xlim, [0, 0], color='k', linestyle='--')
            
            ax1.set(xlabel='$Gray$', xticks=Gray_band, ylabel='$rgb\ distance$', xlim=xlim, ylim=[-0.05 * max_distance, 1.05 * max_distance], title='$from\ predict$')
            ax2.set(xlabel='$Gray$', xticks=Gray_band, ylabel='$rgb\ distance$', xlim=xlim, ylim=[-0.05 * max_distance, 1.05 * max_distance], title='$from\ current$')
            
            ax1.legend()
            ax2.legend()
            fig.tight_layout(pad=1.5)
            fig.savefig('{}.png'.format(filepath), dpi=300)
            plt.close()
            
        means_screen, stds_screen = rgb_distance.mean(axis=0), rgb_distance.std(axis=0)
        
        if self.__log_saved:
            f.write('\nmean = {}\n'.format(means_screen))
            f.write('std  = {}'.format(stds_screen))
            f.close()
            
        return means_screen, stds_screen
        
    def __extract_and_display2(self, mean_std_screens):
        filename = os.path.join(self.res_path, 'all_screens.png')
        
        mean_list = list()
        for mean_std in mean_std_screens:
            mean_list.append(mean_std[0])
        mean_list = numpy.array(mean_list)
        
        if self.__graph_saved:
            fig = plt.figure(figsize=(8, 6))
            ax1 = plt.subplot()
            ax1.plot(range(mean_list.shape[0]), mean_list[:, 0], label='$pred$')
            ax1.plot(range(mean_list.shape[0]), mean_list[:, 1], label='$curr$')
            ax1.set(xlabel='$screen\_idx$', xticks=range(mean_list.shape[0]), ylabel='$mean\ distance$', title='$mean\ distance\ on\ each\ screen$')
            ax1.legend()
            fig.savefig(filename, dpi=300)
            plt.close()
            
        mean = mean_list.mean(axis=0)
        with open('./inertia_sum.log', 'at') as f:
            f.write('{}, screens_tar ={:4d}, screens_ref ={:4d}, pred_mean ={:7.3f}, curr_mean ={:7.3f};\n'.format(time.ctime(), len(self.tar_index_list), len(self.ref_index_list), mean[0], mean[1]))
            
if __name__ == '__main__':
    tar_path = './screens_tar'
    ref_path = './screens_ref'
    
    obj = GTInitial(tar_path, ref_path)
    
    tar_num, ref_num = obj.check_files()
    
    for ref_choose in range(5, ref_num+1):
        print('ref_choose = {}'.format(ref_choose))
        obj.predict_rgb(omega=0, ref_index_list=range(ref_choose), log_saved=False, fig_saved=False)
    
    obj.show_summary()
    




