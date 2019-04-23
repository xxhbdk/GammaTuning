import os
import shutil
import pandas
import numpy
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from collections import OrderedDict


# 完成数据提取
class GammaExcel(object):
    
    def __init__(self, dirname):
        self.dirname = dirname
        self.__get_sheets()
        self.__add_Gray_int()
        self.__add_band_idx()
        self.__get_data_from_bands()
        self.__sort_by_gray()
    
    # 获取所有有效表格 -> 有序字典
    def __get_sheets(self):
        self.filenames = list(os.path.join(self.dirname, filename) for filename in os.listdir(self.dirname) if 'JC' not in filename)
        self.sheets_dict = OrderedDict()
        
        for filename in self.filenames:
            excel = pandas.read_excel(filename, None)
            for key, value in excel.items():
                if 'Check Data' in key:
                    self.sheets_dict[key] = value
    
    # 为所有sheet添加灰度值信息    
    def __add_Gray_int(self):
        for sheet in self.sheets_dict.values():
            Gray_aft_list = list()
            for ele in sheet['Gray']:
                idx_int = ele.find('-')
                Gray_aft_list.append(int(ele[idx_int+1:]))
            sheet['Gray_aft'] = Gray_aft_list
    
    # 为所有sheet添加band索引信息
    def __add_band_idx(self, max_gray=255):
        for sheet in self.sheets_dict.values():
            band_list = list()
            band_index = 0
            
            for idx, row in sheet.iterrows():
                if row['Gray_aft'] == max_gray:
                    band_index += 1
                band_list.append(band_index)
            sheet['band_index'] = band_list
            
            self.bands_num = band_index
        
    # 逐band生成信息{band_idx: [[band_data], [sheet_name]]}
    def __get_data_from_bands(self):
        self.bands_dict = dict()
        
        for sheet_name, sheet in self.sheets_dict.items():
            for idx, band_data in sheet.groupby('band_index'):
                if idx not in self.bands_dict:
                    self.bands_dict[idx] = [[band_data], [sheet_name]]
                else:
                    self.bands_dict[idx][0].append(band_data)
                    self.bands_dict[idx][1].append(sheet_name)
        
    # 根据灰度值, 逐band排序
    def __sort_by_gray(self):
        for band_idx, (bands_data, sheets_name) in self.bands_dict.items():
            for band_data in bands_data:
                band_data.sort_values(by='Gray_aft', inplace=True)
                band_data.reset_index(drop=True, inplace=True)
        


# 完成绘图操作
class GraphDisplay(object):
    
    def __init__(self, gamma_excel_ins):
        self.gamma_excel_ins = gamma_excel_ins         # GammaExcel实例对象
        

        
    # # 循环图片绘制, 一条band一张图片
    def get_graphs(self, path=None, smoothing=True):
        if path:
            if os.path.isdir(path): shutil.rmtree(path)
        else:
            path = './graphs'
            if os.path.isdir(path): shutil.rmtree(path)
            import time
            time.sleep(0.1)
        os.mkdir(path)
        
        # 原值 ####################
        for band_idx, (bands_data, sheets_name) in self.gamma_excel_ins.bands_dict.items():
            fig = plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()
            y_max = 0
            
            for band_data in bands_data:
                x_data = x_data_new = numpy.array(band_data['Gray_aft'])
                y_data = y_data_new = numpy.array(band_data['Actual Lv'])
                
                if y_data.max() > y_max: y_max = y_data.max()
                
                if smoothing:
                    f = interp1d(x_data, y_data, kind='cubic')
                    x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                    y_data_new = f(x_data_new)
                ax1.plot(x_data_new, y_data_new, c='black', linewidth=0.1)
            
            x_data = x_data_new = bands_data[-1]['Gray_aft']
            y_data_min = y_data_min_new = bands_data[-1]['Theory Min Lv']
            y_data_max = y_data_max_new = bands_data[-1]['Theory Max Lv']
            if smoothing:
                f_min = interp1d(x_data, y_data_min, kind='cubic')
                f_max = interp1d(x_data, y_data_max, kind='cubic')
                x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                y_data_min_new = f_min(x_data_new)
                y_data_max_new = f_max(x_data_new)
            ax1.plot(x_data_new, y_data_min_new, label='$Theory\ Min\ Lv$', c='red', linewidth=0.1)
            ax1.plot(x_data_new, y_data_max_new, label='$Theory\ Max\ Lv$', c='orange', linewidth=0.1)
            
            ax1.legend()
            ax1.set(xticks=x_data, xlabel='$Gray$', ylabel='$Lv$', xlim=(-5, 260), ylim=(-0.05 * y_max, y_max * 1.05), title='$band{}$'.format(band_idx))
            for idx, ele in enumerate(y_data_max):
                ax1.plot([x_data[idx], x_data[idx]], [-0.05 * y_max, ele], 'b--', linewidth=0.3)
                ax1.plot([-5, x_data[idx]], [ele, ele], 'b--', linewidth=0.3)
                
            for idx, ele in enumerate(y_data_min):
                ax1.plot([-5, x_data[idx]], [ele, ele], 'b--', linewidth=0.3)
            
            fig.savefig(os.path.join(path, 'ori_band{}.png'.format(band_idx)), dpi=500)
            plt.close()

        #####################################################################################################
        # 归一化 #######################
        
        for band_idx, (bands_data, sheets_name) in self.gamma_excel_ins.bands_dict.items():
            fig = plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()
            y_max = 0
            
            for band_data in bands_data:
                x_data = x_data_new = numpy.array(band_data['Gray_aft'])
                y_data = y_data_new = (band_data['Actual Lv'] - bands_data[-1]['Theory Min Lv']) / (band_data['Theory Max Lv'] - band_data['Theory Min Lv'])
                
                if y_data.max() > y_max: y_max = y_data.max()
                
                if smoothing:
                    f = interp1d(x_data, y_data, kind='cubic')
                    x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                    y_data_new = f(x_data_new)
                ax1.plot(x_data_new, y_data_new, c='black', linewidth=0.5)
            
            x_data = x_data_new = bands_data[-1]['Gray_aft']
            y_data_min = y_data_min_new = (bands_data[-1]['Theory Min Lv'] - bands_data[-1]['Theory Min Lv']) / (band_data['Theory Max Lv'] - band_data['Theory Min Lv'])
            y_data_max = y_data_max_new = (bands_data[-1]['Theory Max Lv'] - bands_data[-1]['Theory Min Lv']) / (band_data['Theory Max Lv'] - band_data['Theory Min Lv'])
            if smoothing:
                f_min = interp1d(x_data, y_data_min, kind='cubic')
                f_max = interp1d(x_data, y_data_max, kind='cubic')
                x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                y_data_min_new = f_min(x_data_new)
                y_data_max_new = f_max(x_data_new)
            ax1.plot(x_data_new, y_data_min_new, label='$Theory\ Min\ Lv$', c='red', linewidth=0.5)
            ax1.plot(x_data_new, y_data_max_new, label='$Theory\ Max\ Lv$', c='orange', linewidth=0.5)
            
            ax1.legend()
            ax1.grid(which='major', axis='x', linestyle='--', color='blue', linewidth=0.5)
            ax1.set(xticks=x_data, xlabel='$Gray$', ylabel='$Lv\_norm$', title='$band{}$'.format(band_idx))
            
            fig.savefig(os.path.join(path, 'norm_band{}.png'.format(band_idx)), dpi=500)
            plt.close()


        #####################################################################################################
        
        # 总图 ##############
        fig = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot()
        y_max = 0
        
        for band_idx, (bands_data, sheets_name) in self.gamma_excel_ins.bands_dict.items():
            
            for band_data in bands_data:
                x_data = x_data_new = numpy.array(band_data['Gray_aft'])
                y_data = y_data_new = numpy.array(band_data['Actual Lv'])
                
                if y_data.max() > y_max: y_max = y_data.max()
                
                if smoothing:
                    f = interp1d(x_data, y_data, kind='cubic')
                    x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                    y_data_new = f(x_data_new)
                ax1.plot(x_data_new, y_data_new, c='black', linewidth=0.1)
            
            x_data = x_data_new = bands_data[-1]['Gray_aft']
            y_data_min = y_data_min_new = bands_data[-1]['Theory Min Lv']
            y_data_max = y_data_max_new = bands_data[-1]['Theory Max Lv']
            if smoothing:
                f_min = interp1d(x_data, y_data_min, kind='cubic')
                f_max = interp1d(x_data, y_data_max, kind='cubic')
                x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                y_data_min_new = f_min(x_data_new)
                y_data_max_new = f_max(x_data_new)
            ax1.plot(x_data_new, y_data_min_new, label='$Theory\ Min\ Lv$', c='red', linewidth=0.1)
            ax1.plot(x_data_new, y_data_max_new, label='$Theory\ Max\ Lv$', c='orange', linewidth=0.1)

        ax1.set(xticks=x_data, xlabel='$Gray$', ylabel='$Lv$', xlim=(-5, 260), ylim=(-0.05 * y_max, 1.05 * y_max), title='$all\ bands$')
            
        handles, labels = ax1.get_legend_handles_labels()
        label_handle_dict = OrderedDict(zip(labels, handles))
        ax1.legend(label_handle_dict.values(), label_handle_dict.keys())
        
        fig.savefig(os.path.join(path, 'all_bands.png'.format(band_idx)), dpi=500)
        plt.close()
        
        #################################################################################################

        # 差值
        for band_idx, (bands_data, sheets_name) in self.gamma_excel_ins.bands_dict.items():
            fig = plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()
            y_max = 0
            
            for band_data in bands_data:
                x_data = x_data_new = numpy.array(band_data['Gray_aft'])
                y_data = y_data_new = band_data['Actual Lv'] - bands_data[-1]['Theory Min Lv']   ##########

                if y_data.max() > y_max: y_max = y_data.max()
                
                if smoothing:
                    f = interp1d(x_data, y_data, kind='cubic')
                    x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                    y_data_new = f(x_data_new)
                ax1.plot(x_data_new, y_data_new, c='black', linewidth=0.5)
            
            x_data = x_data_new = bands_data[-1]['Gray_aft']
            y_data_min = y_data_min_new = bands_data[-1]['Theory Min Lv'] - bands_data[-1]['Theory Min Lv']       ##########
            y_data_max = y_data_max_new = bands_data[-1]['Theory Max Lv'] - bands_data[-1]['Theory Min Lv']     ##########
            if smoothing:
                f_min = interp1d(x_data, y_data_min, kind='cubic')
                f_max = interp1d(x_data, y_data_max, kind='cubic')
                x_data_new = numpy.linspace(x_data.min(), x_data.max(), 500)
                y_data_min_new = f_min(x_data_new)
                y_data_max_new = f_max(x_data_new)
            ax1.plot(x_data_new, y_data_min_new, label='$Theory\ Min\ Lv$', c='red', linewidth=0.5)
            ax1.plot(x_data_new, y_data_max_new, label='$Theory\ Max\ Lv$', c='orange', linewidth=0.5)
            
            ax1.legend()
            ax1.grid(which='major', axis='x', linestyle='--', color='blue', linewidth=0.5)
            ax1.set(xticks=x_data, xlabel='$Gray$', ylabel='$Lv\ -\ Theory\ Min\ Lv$', title='$band{}$'.format(band_idx))
            
            fig.savefig(os.path.join(path, 'diff_band{}.png'.format(band_idx)), dpi=500)
            plt.close()

        
        
        
if __name__ == '__main__':
    gamma_ins = GammaExcel('./Gamma_excel2')
    graph_ins = GraphDisplay(gamma_ins)
    graph_ins.get_graphs()





