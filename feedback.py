# 数据反馈及绘图展示模块

import time
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from preprocessing import DataPreprocessing
from programming import Solver


class FeedBack(object):
    
    def __init__(self, path_ref, path_tar):
        self.path_ref = path_ref      # 参考屏路径
        self.path_tar = path_tar      # 目标屏路径
        
    def load_all_ref(self):
        self.data_ins = DataPreprocessing(self.path_ref, self.path_tar)
        self.data_ins.load_all_ref()
        
    def load_assigned_tar(self, tar_idx=0):
        self.__tar_idx = tar_idx      # 记录屏幕编号
        self.data_ins.load_assigned_tar(tar_idx)
        
    def load_data(self, tar_idx=0):
        '''
        tar_index: 目标屏编号
        '''
        self.__tar_idx = tar_idx      # 记录屏幕编号
        self.load_all_ref()
        self.load_assigned_tar(tar_idx)
    
    def get_A_D(self, Lmax_idx=0, Gray_idx=0, status='stop'):
        '''
        获取用于计算的A矩阵与D矢量
        '''
        A = self.data_ins.get_A_from_ref(Lmax_idx, Gray_idx, status)
        D = self.data_ins.get_D_from_tar(Lmax_idx, Gray_idx, status)
        return (A, D)
    
    def show_diff(self):
        '''
        均匀初值、加权初值、现有初值与现有终值的差值(2范数)展示
        需要标记各band上的均值与方差
        优化方向: Lmax由高向低, Gray由高向低
        '''
        solver = Solver()
        f = open('./res_100/screen_{}.txt'.format(self.__tar_idx), 'wt')
        f.write('优化方向: Lmax由高向低, Gray由高向低\n')
        f.write('mean_dist: 参考屏(r, g, b)均匀初值与目标屏(r, g, b)现有终值之间的距离\n')
        f.write('weight_dist: 参考屏(r, g, b)加权初值与目标屏(r, g, b)现有终值之间的距离\n')
        f.write('mean_dist: 目标屏(r, g, b)现有初值与目标屏(r, g, b)现有终值之间的距离\n\n')
        
        fig = plt.figure(figsize=(9, 12))
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        
        A_init, D_init = self.get_A_D(0, 0, 'init')
        w_init = numpy.array([1 / A_init.shape[1]] * A_init.shape[1]).reshape(-1, 1)     # 第一个初值 - 权重均匀初始化
        
        mean_list = list()
        weight_list = list()
        curr_list = list()
        for Lmax_idx in range(5, -1, -1):
            w = w_init
            
            mean_band = list()
            weight_band = list()
            curr_band = list()
            
            f.write('************** band {} ****************\n'.format(Lmax_idx))
            for Gray_idx in range(32, -1, -1):
                A_init, D_init = self.get_A_D(Lmax_idx, Gray_idx, 'init')
                A_stop, D_stop = self.get_A_D(Lmax_idx, Gray_idx, 'stop')
                
                mean_init = numpy.around(A_stop.mean(axis=1))                      # 均匀初值
                weight_init = numpy.around(numpy.matmul(A_stop, w)).reshape(-1)    # 加权初值
                curr_init = D_init.reshape(-1)                                     # 现有初值
                curr_stop = D_stop.reshape(-1)                                     # 现有终值
                
                mean_diff = numpy.linalg.norm(mean_init - curr_stop)               ###
                weight_diff = numpy.linalg.norm(weight_init - curr_stop)           ###
                curr_diff = numpy.linalg.norm(curr_init - curr_stop)               ###
                
                mean_list.append(mean_diff)
                weight_list.append(weight_diff)
                curr_list.append(curr_diff)
                
                mean_band.append(mean_diff)
                weight_band.append(weight_diff)
                curr_band.append(curr_diff)
                
                f.write('mean_dist = {:7.3f}; weight_dist = {:7.3f}; curr_dist = {:7.3f}\n'.format(mean_diff, weight_diff, curr_diff))
                w = solver.solve2norm(A_stop, D_stop)
                
                if Gray_idx == 32:
                    w_init = w
                    
            mean_band.reverse()
            weight_band.reverse()
            curr_band.reverse()
            
            ax1.plot(mean_band, label='$band{}$'.format(Lmax_idx))
            ax2.plot(weight_band, label='$band{}$'.format(Lmax_idx))
            ax3.plot(curr_band, label='$band{}$'.format(Lmax_idx))
            
        max_diff = max(max(weight_list), max(curr_list))
        ax1.set(xlabel='$Gray\_idx$', ylabel='$mean\ distance$')
        ax2.set(ylim=(-0.05 * max_diff, 1.05 * max_diff), xlabel='$Gray\_idx$', ylabel='$weight\ distance$')
        ax3.set(ylim=(-0.05 * max_diff, 1.05 * max_diff), xlabel='$Gray\_idx$', ylabel='$current\ distance$')
        ax1.legend()
        ax2.legend()
        ax3.legend()
        fig.tight_layout(pad=1.5)    
        fig.savefig('./res_100/screen_{}.png'.format(self.__tar_idx), dpi=500)
        plt.close()
        
                    
        f.write('\n>>>过滤前<<<\n')
        f.write('均匀初值:\n')
        f.write('均值: {:9.3f}; 方差: {:9.3f}\n'.format(numpy.mean(mean_list), numpy.std(mean_list)))
        f.write('加权初值:\n')
        f.write('均值: {:9.3f}; 方差: {:9.3f}\n'.format(numpy.mean(weight_list), numpy.std(weight_list)))
        f.write('现有初值:\n')
        f.write('均值: {:9.3f}; 方差: {:9.3f}\n'.format(numpy.mean(curr_list), numpy.std(curr_list)))
        f.write('\n>>>过滤后<<<\n')
        mean_list_aft = list(val for val in mean_list if val > 1.733)
        weight_list_aft = list(val for val in weight_list if val > 1.733)
        curr_list_aft = list(val for val in curr_list if val > 1.733)
        f.write('均匀初值:\n')
        f.write('均值: {:9.3f}; 方差: {:9.3f}\n'.format(numpy.mean(mean_list_aft), numpy.std(mean_list_aft)))
        f.write('加权初值:\n')
        f.write('均值: {:9.3f}; 方差: {:9.3f}\n'.format(numpy.mean(weight_list_aft), numpy.std(weight_list_aft)))
        f.write('现有初值:\n')
        f.write('均值: {:9.3f}; 方差: {:9.3f}\n'.format(numpy.mean(curr_list_aft), numpy.std(curr_list_aft)))
        f.close()
        
        # return ((numpy.mean(weight_list_aft), numpy.std(weight_list_aft)), (numpy.mean(curr_list_aft), numpy.std(curr_list_aft)))
        return ((numpy.mean(weight_list), numpy.std(weight_list)), (numpy.mean(curr_list), numpy.std(curr_list)))
        
    
    
if __name__ == '__main__':
    path_ref = './ref_new'
    path_tar = './tar_new'
    
    fb = FeedBack(path_ref, path_tar)
    fb.load_all_ref()
    print('start at: {}'.format(time.ctime()))
    
    weight_mean_list = list()
    weight_std_list = list()
    curr_mean_list = list()
    curr_std_list = list()
    for i in range(30):
        print('the {}th screen!'.format(i))
        fb.load_assigned_tar(i)
        res = fb.show_diff()
        weight_mean_list.append(res[0][0])
        weight_std_list.append(res[0][1])
        curr_mean_list.append(res[1][0])
        curr_std_list.append(res[1][1])
    
    fig = plt.figure(figsize=(8, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    ax1.plot(weight_mean_list, label='$weight$')
    ax1.plot(curr_mean_list, label='$current$')
    
    ax2.plot(weight_std_list, label='$weight$')
    ax2.plot(curr_std_list, label='$current$')
    
    ax1.set(xlabel='$screen\ number$', ylabel='$mean\ distance$', xticks=range(30))
    ax2.set(xlabel='$screen\ number$', ylabel='$std\ of\ distance$', xticks=range(30))
    
    ax1.legend()
    ax2.legend()
    
    fig.tight_layout(pad=1.5)
    fig.savefig('./res_100/total.png', dpi=500)
    plt.close()
    
    print('end at: {}'.format(time.ctime()))
    
    
        

