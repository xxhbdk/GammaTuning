# 完成A矩阵的计算
# 完成线性模型的绘图检查(并采用线性拟合)

import os

import numpy
import pandas
from matplotlib import pyplot as plt


# 完成数值的转换
class Transform(object):
    
    def __init__(self, G_VREG1=6.5, G_VREF1=1, G_VT_REG_V=0x12):
        self.G_VREG1 = G_VREG1
        self.G_VREF1 = G_VREF1
        self.G_VT_REG_V = G_VT_REG_V
        
        self.G_VTUPP = G_VREF1 + (G_VREG1 - G_VREF1) / 2047 * (2047 - G_VT_REG_V)

        
    def calc_vol_by_reg(self, Gray, RegValue, LastGrayVol, GrayV0):
        if Gray == 255:
            nVol = self.G_VREF1 + (self.G_VREG1 - self.G_VREF1) / 2047 * (2047 - RegValue)
        elif Gray > 7:
            nVol = LastGrayVol + (self.G_VTUPP - LastGrayVol) / 511 * (511 - RegValue)
        elif Gray > 0:
            nVol = LastGrayVol + (self.G_VREG1 - LastGrayVol) / 511 * (511 - RegValue)
        elif Gray == 0:
            nVol = GrayV0
        else:
            raise Exception('>>> Gray = "{}" out of range <<<'.format(Gray))
        
        return nVol
        
        
    def calc_uv_by_xy(self, x, y):
        u = 4 * x / (-2 * x + 12 * y + 3)
        v = 9 * y / (-2 * x + 12 * y + 3)
        
        return u, v
        
        

# 完成指定路径下csv文件的解析
class FileParser(object):

    def __init__(self, path):
        self.path = path
        
        
    def load_files(self):
        self.filename_list_ori = list(filename for filename in os.listdir(self.path) if filename.endswith('.csv'))
        self.filename_list_tra = self.__sorted_filenames(self.filename_list_ori)
        
        self.csv_list = list(pandas.read_csv(os.path.join(self.path, filename), names=['x', 'y', 'Lv', 'Reg_r', 'Reg_g', 'Reg_b']) for filename in self.filename_list_tra)
        self.__add_mean()
        
        
    def __add_mean(self):
        for i in range(len(self.csv_list)):
            # 找到中心
            row_mean_ori = self.csv_list[i].mean()
            
            csv_filter = self.csv_list[i][(self.csv_list[i]['Reg_r'] == row_mean_ori['Reg_r']) & (self.csv_list[i]['Reg_g'] == row_mean_ori['Reg_g']) & (self.csv_list[i]['Reg_b'] == row_mean_ori['Reg_b'])]
            if csv_filter.size == 0:
                raise Exception('>>> can not find the mean rgb({}, {}, {}) in file named with "{}" <<<'.format(*row_mean_ori.tolist()[3:], self.filename_list_tra[i]))
            
            row_mean_tra = csv_filter.mean()
            self.csv_list[i] = self.csv_list[i].append(row_mean_tra, ignore_index=True)


    def convert2norm(self, GrayV0_list=None, **kwargs):
        '''
        GrayV0_list = [GrayV0_band1, GrayV0_band2, GrayV0_band3, ...]
        '''
        if GrayV0_list is None:
            GrayV0_list = [6.5] * 17
    
        if not hasattr(self, 'csv_list'):
            self.load_files()
        
        self.__convert2norm(GrayV0_list, **kwargs)
        
        
    def __convert2norm(self, GrayV0_list, **kwargs):
        Gray_list = list(int(filename[filename.find('W') + 1:filename.find('.')]) for filename in self.filename_list_tra)
        band_list = list(int(filename[filename.find('band') + 4:filename.find('_')]) - 1 for filename in self.filename_list_tra)
        
        self.csv_list_new = list()
        
        transform = Transform(**kwargs)
        # 仔细检查 #######################
        Vol_r, Vol_g, Vol_b = None, None, None
        for band_num, Gray, csv in zip(band_list, Gray_list, self.csv_list):
            u_list, v_list, Vol_r_list, Vol_g_list, Vol_b_list = list(), list(), list(), list(), list()
            
            for row_index, row in csv.iterrows():
                u, v = transform.calc_uv_by_xy(row['x'], row['y'])
                u_list.append(u)
                v_list.append(v)
                
                Vol_r2 = transform.calc_vol_by_reg(Gray, row['Reg_r'], Vol_r, GrayV0_list[band_num])
                Vol_g2 = transform.calc_vol_by_reg(Gray, row['Reg_g'], Vol_g, GrayV0_list[band_num])
                Vol_b2 = transform.calc_vol_by_reg(Gray, row['Reg_b'], Vol_b, GrayV0_list[band_num])
                Vol_r_list.append(Vol_r2)
                Vol_g_list.append(Vol_g2)
                Vol_b_list.append(Vol_b2)
            else:
                Vol_r = Vol_r2
                Vol_g = Vol_g2
                Vol_b = Vol_b2
                
            csv_new = pandas.DataFrame({'u': u_list, 'v': v_list, 'Lv': csv['Lv'], 'Vol_r': Vol_r_list, 'Vol_g': Vol_g_list, 'Vol_b': Vol_b_list})
            self.csv_list_new.append(csv_new)
            
            
    def __sorted_filenames(self, filename_list_ori):
        band2Gray = dict()
        
        for filename in filename_list_ori:
            band_num, Gray_tab = filename.split('_')
            if band_num not in band2Gray:
                band2Gray[band_num] = [Gray_tab]
            else:
                band2Gray[band_num].append(Gray_tab)
        
        self.Gray_list = list()
        for Gray_list in band2Gray.values():
            Gray_list.sort(key=lambda item: int(item[item.find('W') + 1:item.find('.')]), reverse=True)
            if len(Gray_list) > len(self.Gray_list):
                self.Gray_list = list(int(item[item.find('W') + 1:item.find('.')]) for item in Gray_list)
            
        band_list = sorted(list(band2Gray.keys()))
        
        filename_list_tra = list()
        for band_num in band_list:
            filename_list_tra.extend('{}_{}'.format(band_num, Gray) for Gray in band2Gray[band_num])
            
        return filename_list_tra



class CalcAMatrix(object):
    
    def __init__(self, path):
        self.path = path
        
    
    def feed_params(self, **kwargs):
        '''
        G_VREG1 = 6.5
        G_VREF1 = 1
        G_VT_REG_V = 0x12
        band_num = 17
        GrayV0_list = [6.5] * self.band_num
        Gray_list = [Gray_list_from_filenames]
        '''
        self.G_VREG1 = kwargs['G_VREG1'] if 'G_VREG1' in kwargs else 6.5
        self.G_VREF1 = kwargs['G_VREF1'] if 'G_VREF1' in kwargs else 1
        self.G_VT_REG_V = kwargs['G_VT_REG_V'] if 'G_VT_REG_V' in kwargs else 0x12
        self.band_num = kwargs['band_num'] if 'band_num' in kwargs else 17
        self.GrayV0_list = kwargs['GrayV0_list'] if 'GrayV0_list' in kwargs else [6.5] * self.band_num
        self.Gray_list = sorted(kwargs['Gray_list'], reverse=True) if 'Gray_list' in kwargs else None
        
        
    def __make_trans(self):
        if not hasattr(self, 'G_VREG1'):
            self.feed_params()
            
        trans = FileParser(self.path)
        
        trans.load_files()
        
        trans.convert2norm(GrayV0_list=self.GrayV0_list, G_VREG1=self.G_VREG1, G_VREF1=self.G_VREF1, G_VT_REG_V=self.G_VT_REG_V)
        
        self.filename_list = trans.filename_list_tra
        self.csv_list = trans.csv_list_new
        
        if not self.Gray_list:
            self.Gray_list = trans.Gray_list
        
        
    def get_AMatrix(self, A_saved=True, A_filename=None, epsilon=1.e-9):
        '''
        计算指定band上所有绑点的A矩阵, 缺失则填入前一条band上对应灰阶的A矩阵
        '''
        if not hasattr(self, 'filename_list'):
            self.__make_trans()
            
        self.AMatrix_list = list([None] * len(self.Gray_list) for i in range(self.band_num))
            
        for filename, csv in zip(self.filename_list, self.csv_list):
            band_idx, Gray_idx = self.__get_loc(filename)
            AMatrix = self.__calc_AMatrix(csv, epsilon)
            self.AMatrix_list[band_idx][Gray_idx] = AMatrix
            
        self.__fill_AMatrix()                     # 填充缺失矩阵
            
        if A_saved:
            if A_filename is None: A_filename = './Amat.dat'
            
            self.__save_AMatrix(A_filename)
            
            
    def __get_loc(self, filename):
        band_num = int(filename[filename.find('band') + 4:filename.find('_')])
        Gray_num = int(filename[filename.find('W') + 1:filename.find('.')])
        
        return band_num - 1, self.Gray_list.index(Gray_num)
        
        
    def __calc_AMatrix(self, csv, epsilon):
        row_last = csv.iloc[-1, :]
        delta_csv = csv - row_last
        
        delta_r = delta_csv['Vol_r'].values[:-1].reshape(-1, 1)
        delta_g = delta_csv['Vol_g'].values[:-1].reshape(-1, 1)
        delta_b = delta_csv['Vol_b'].values[:-1].reshape(-1, 1)
        
        A = numpy.hstack((delta_r, delta_g, delta_b))
        D_u = delta_csv['u'].values[:-1].reshape(-1, 1)
        D_v = delta_csv['v'].values[:-1].reshape(-1, 1)
        D_Lv = delta_csv['Lv'].values[:-1].reshape(-1, 1)
        
        W_u = self.__calc_W(A, D_u, epsilon)
        W_v = self.__calc_W(A, D_v, epsilon)
        W_Lv = self.__calc_W(A, D_Lv, epsilon)
        
        AMatrix = numpy.hstack([W_Lv, W_u, W_v]).T
        
        return AMatrix
        
        
    def __fill_AMatrix(self):
        for band_idx in range(self.band_num):
            for Gray_idx in range(len(self.Gray_list)):
                AMatrix = self.AMatrix_list[band_idx][Gray_idx]
                
                if AMatrix is None:
                    band_seq = self.__get_band_seq(band_idx)          # 纵向填充序列
                    for band_idx_sub in band_seq:
                        AMatrix_tmp = self.AMatrix_list[band_idx_sub][Gray_idx]
                        
                        if AMatrix_tmp is not None:
                            self.AMatrix_list[band_idx][Gray_idx] = AMatrix_tmp
                            break
                    else:
                        Gray_seq2 = self.__get_band_seq2(Gray_idx)    # 横向填充序列
                        for Gray_idx_sub in Gray_seq2:
                            AMatrix_tmp = self.AMatrix_list[band_idx][Gray_idx_sub]
                            
                            if AMatrix_tmp is not None:
                                self.AMatrix_list[band_idx][Gray_idx] = AMatrix_tmp
                                break
                        else:
                            raise Exception('>>> something confusing happened on (band_idx, Gray_idx) = ({}, {}) <<<'.format(band_idx, Gray_idx))


    def __get_band_seq(self, band_idx):
        seq_bef = list(reversed(range(band_idx)))
        seq_aft = list(range(band_idx + 1, self.band_num))
        
        return seq_bef + seq_aft
        
    
    def __get_band_seq2(self, Gray_idx):
        seq_bef = list(reversed(range(Gray_idx)))
        seq_aft = list(range(Gray_idx + 1, len(self.Gray_list)))
        
        return seq_bef + seq_aft
        
        
    def __calc_W(self, A, D, epsilon):
        item1 = numpy.matmul(A.T, A)
        item2 = numpy.identity(A.shape[1]) * epsilon
        item3 = numpy.matmul(A.T, D)
        
        W = numpy.matmul(numpy.linalg.inv(item2 + item1), item3)
        
        return W
        
        
    def __save_AMatrix(self, filename):
        with open(filename, 'wt') as f:            
            for band_idx in range(self.band_num):
                for Gray_idx in range(len(self.Gray_list)):
                    AMatrix = self.AMatrix_list[band_idx][Gray_idx]
                    AMatrix_list = AMatrix.reshape(-1).tolist()
                    line = '{} {} {}\n'.format(band_idx + 1, self.Gray_list[Gray_idx], ' '.join('{:.5f}'.format(ele) for ele in AMatrix_list))
                    f.write(line)
                    
                    
                    
class LinearCheck(object):
    
    def __init__(self, path):
        self.path = path
        
        
    def feed_params(self, **kwargs):
        '''
        G_VREG1 = 6.5
        G_VREF1 = 1
        G_VT_REG_V = 0x12
        band_num = 17
        GrayV0_list = [6.5] * self.band_num
        '''
        self.G_VREG1 = kwargs['G_VREG1'] if 'G_VREG1' in kwargs else 6.5
        self.G_VREF1 = kwargs['G_VREF1'] if 'G_VREF1' in kwargs else 1
        self.G_VT_REG_V = kwargs['G_VT_REG_V'] if 'G_VT_REG_V' in kwargs else 0x12
        self.band_num = kwargs['band_num'] if 'band_num' in kwargs else 17
        self.GrayV0_list = kwargs['GrayV0_list'] if 'GrayV0_list' in kwargs else [6.5] * self.band_num
        
            
    def __make_trans(self):
        if not hasattr(self, 'G_VREG1'):
            self.feed_params()
        
        trans = FileParser(self.path)
        
        trans.load_files()
        
        trans.convert2norm(GrayV0_list=self.GrayV0_list, G_VREG1=self.G_VREG1, G_VREF1=self.G_VREF1, G_VT_REG_V=self.G_VT_REG_V)
        
        self.filename_list = trans.filename_list_tra
        self.csv_list_old = trans.csv_list
        self.csv_list_new = trans.csv_list_new
        
        
    def get_Graph(self, fig_saved=True, epsilon=1.e-9):
        if not hasattr(self, 'filename_list'):
            self.__make_trans()
            
        for filename, csv_old, csv_new in zip(self.filename_list, self.csv_list_old, self.csv_list_new):
            gap = (csv_old.shape[0] - 1) / 3
            
            r_start = self.__get_start(csv_old, 'Reg_r')
            g_start = self.__get_start(csv_old, 'Reg_g')
            b_start = self.__get_start(csv_old, 'Reg_b')
            
            if not (r_start < g_start < b_start):
                raise Exception('>>> does not satisfy our convention: "r_start({}) < g_start({}) < b_start({})" <<<'.format(r_start, g_start, b_start))
            
            start_list = [r_start, g_start, b_start]
            self.__get_Graph(csv_old, csv_new, start_list, filename, fig_saved, epsilon)
            
            
    def __get_Graph(self, csv_old, csv_new, start_list, filename, fig_saved, epsilon):
        old_column_r = csv_old['Reg_r'].values.reshape(-1, 1)
        old_column_g = csv_old['Reg_g'].values.reshape(-1, 1)
        old_column_b = csv_old['Reg_b'].values.reshape(-1, 1)
        old_column_Lv = csv_old['Lv'].values.reshape(-1, 1)
        old_column_x = csv_old['x'].values.reshape(-1, 1)
        old_column_y = csv_old['y'].values.reshape(-1, 1)
        
        new_column_r = csv_new['Vol_r'].values.reshape(-1, 1)
        new_column_g = csv_new['Vol_g'].values.reshape(-1, 1)
        new_column_b = csv_new['Vol_b'].values.reshape(-1, 1)
        new_column_Lv = csv_new['Lv'].values.reshape(-1, 1)
        new_column_u = csv_new['u'].values.reshape(-1, 1)
        new_column_v = csv_new['v'].values.reshape(-1, 1)
        
        old_data_r = numpy.hstack([old_column_r[start_list[0]:start_list[1], :], old_column_Lv[start_list[0]:start_list[1], :], old_column_x[start_list[0]:start_list[1], :], old_column_y[start_list[0]:start_list[1], :]])
        old_data_g = numpy.hstack([old_column_g[start_list[1]:start_list[2], :], old_column_Lv[start_list[1]:start_list[2], :], old_column_x[start_list[1]:start_list[2], :], old_column_y[start_list[1]:start_list[2], :]])
        old_data_b = numpy.hstack([old_column_b[start_list[2]:-1, :], old_column_Lv[start_list[2]:-1, :], old_column_x[start_list[2]:-1, :], old_column_y[start_list[2]:-1, :]])
        
        new_data_r = numpy.hstack([new_column_r[start_list[0]:start_list[1], :], new_column_Lv[start_list[0]:start_list[1], :], new_column_u[start_list[0]:start_list[1], :], new_column_v[start_list[0]:start_list[1], :]])
        new_data_g = numpy.hstack([new_column_g[start_list[1]:start_list[2], :], new_column_Lv[start_list[1]:start_list[2], :], new_column_u[start_list[1]:start_list[2], :], new_column_v[start_list[1]:start_list[2], :]])
        new_data_b = numpy.hstack([new_column_b[start_list[2]:-1, :], new_column_Lv[start_list[2]:-1, :], new_column_u[start_list[2]:-1, :], new_column_v[start_list[2]:-1, :]])
        
        if fig_saved:
            self.__plot(old_data_r, old_data_g, old_data_b, epsilon, 'old', filename)
            self.__plot(new_data_r, new_data_g, new_data_b, epsilon, 'new', filename)
            
        
    def __plot(self, data_r, data_g, data_b, epsilon, tab, filename):
        r_min, r_max = data_r[:, 0].min(), data_r[:, 0].max()
        g_min, g_max = data_g[:, 0].min(), data_g[:, 0].max()
        b_min, b_max = data_b[:, 0].min(), data_b[:, 0].max()
        
        fig = plt.figure(figsize=(30, 15))
        
        ax1 = self.__subplot(data_r[:, 0:1], data_r[:, 1:2], 1, epsilon, r_min, r_max)
        ax4 = self.__subplot(data_r[:, 0:1], data_r[:, 2:3], 4, epsilon, r_min, r_max)
        ax7 = self.__subplot(data_r[:, 0:1], data_r[:, 3:4], 7, epsilon, r_min, r_max)
        ax2 = self.__subplot(data_g[:, 0:1], data_g[:, 1:2], 2, epsilon, g_min, g_max)
        ax5 = self.__subplot(data_g[:, 0:1], data_g[:, 2:3], 5, epsilon, g_min, g_max)
        ax8 = self.__subplot(data_g[:, 0:1], data_g[:, 3:4], 8, epsilon, g_min, g_max)
        ax3 = self.__subplot(data_b[:, 0:1], data_b[:, 1:2], 3, epsilon, b_min, b_max)
        ax6 = self.__subplot(data_b[:, 0:1], data_b[:, 2:3], 6, epsilon, b_min, b_max)
        ax9 = self.__subplot(data_b[:, 0:1], data_b[:, 3:4], 9, epsilon, b_min, b_max)
        
        if tab == 'old':
            ax1.set(xlabel='$Reg\_r$', ylabel='$Lv$')
            ax4.set(xlabel='$Reg\_r$', ylabel='$x$')
            ax7.set(xlabel='$Reg\_r$', ylabel='$y$')
            ax2.set(xlabel='$Reg\_g$', ylabel='$Lv$')
            ax5.set(xlabel='$Reg\_g$', ylabel='$x$')
            ax8.set(xlabel='$Reg\_g$', ylabel='$y$')
            ax3.set(xlabel='$Reg\_b$', ylabel='$Lv$')
            ax6.set(xlabel='$Reg\_b$', ylabel='$x$')
            ax9.set(xlabel='$Reg\_b$', ylabel='$y$')
        else:
            ax1.set(xlabel='$Vol\_r$', ylabel='$Lv$')
            ax4.set(xlabel='$Vol\_r$', ylabel='$u$')
            ax7.set(xlabel='$Vol\_r$', ylabel='$v$')
            ax2.set(xlabel='$Vol\_g$', ylabel='$Lv$')
            ax5.set(xlabel='$Vol\_g$', ylabel='$u$')
            ax8.set(xlabel='$Vol\_g$', ylabel='$v$')
            ax3.set(xlabel='$Vol\_b$', ylabel='$Lv$')
            ax6.set(xlabel='$Vol\_b$', ylabel='$u$')
            ax9.set(xlabel='$Vol\_b$', ylabel='$v$')
        
        fig.tight_layout()
        fig.savefig(os.path.join(self.path, '{}_{}.png'.format(os.path.splitext(filename)[0], tab)), dpi=200)
        plt.close()
        
        
    def __subplot(self, X, Y, idx, epsilon, x_min, x_max):
        ax0 = plt.subplot(3, 3, idx)
        
        ax0.scatter(X, Y, s=10, c='red', marker='o')
        
        W = self.linear_fitting(X, Y, epsilon)
        x1, x2 = x_min, x_max
        y1, y2 = W[0, 0] * x1 + W[1, 0], W[0, 0] * x2 + W[1, 0]
        
        ax0.plot([x1, x2], [y1, y2], 'k--', alpha=0.8, label='$y\ =\ {}x{}$'.format('{:.5f}'.format(W[0, 0]), '{:+.5f}'.format(W[1, 0])))
        ax0.legend()
        
        return ax0
        
        
    def linear_fitting(self, X, Y, epsilon=1.e-9):
        '''
        线性拟合
        '''
        rows, cols = X.shape
        X_new = numpy.hstack([X, numpy.ones([rows, 1])])
        
        item0 = numpy.identity(cols) * epsilon + numpy.matmul(X_new.T, X_new)
        item1 = numpy.linalg.inv(item0)
        item2 = numpy.matmul(X_new.T, Y)
        
        W = numpy.matmul(item1, item2)
        
        return W
            
            
    def __get_start(self, csv, column_name):
        origin = csv.iloc[-1, :]                          # 原点坐标信息
        
        column_names_all = ['Reg_r', 'Reg_g', 'Reg_b']
        column_names_out = list(set(column_names_all) - {column_name})
        
        for row_idx, row in csv.iterrows():
            if row[column_name] != origin[column_name] and row[column_names_out[0]] == origin[column_names_out[0]] and row[column_names_out[1]] == origin[column_names_out[1]]:
                return row_idx
        else:
            raise Exception('>>> lack of "{}" <<<'.format(column_name))


        
        
        
if __name__ == '__main__':
    obj = CalcAMatrix('./540S')
    # band_num = 3
    # Gray_list = [255, 239, 207, 143, 111, 95, 79, 63, 55, 39, 31, 15, 7, 5, 3, 1]
    # obj.feed_params(band_num=3, Gray_list=Gray_list)
    obj.get_AMatrix()
    
    obj2 = LinearCheck('./540S')
    obj2.get_Graph()

