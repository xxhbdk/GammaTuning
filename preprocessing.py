# 数据预处理模块

# 1. 根据最大灰阶添加band索引信息band_index
# 2. 添加rgb的十进制信息(r_Dec, g_Dec, b_Dec)
# 3. 添加差值信息Lv - Lt
# 4. 提取每块屏的初始状态与终止状态
# 5. 提取每块屏的计算信息1((Gray, Lmax), (r_Dec, g_Dec, b_Dec))

import os
import numpy
import pandas

class ExcelDealing(object):
    
    def __init__(self, filename, strainer='W-'):
        self.filename = filename       # excel文件名
        self.strainer = strainer       # sheet过滤字段
    
    # 获取excel下的指定sheet
    def get_sheet(self):
        self.__get_sheet()
        self.__add_band_index()        # 向sheet中添加band索引信息
        self.__add_Lmax()              # 向sheet中添加Lmax信息
        self.__add_rgb_Dec()           # 向sheet中添加10进制rgb信息
        self.__add_Lv_diff()           # 添加Lv与Lt的差值信息
        
        return self.sheet
    
    # 获取初始状态与终止状态
    def get_status(self):
        self.get_sheet()
        self.__get_status()
        
        return self.sheet_init, self.sheet_stop
        
    def __get_sheet(self):
        excel = pandas.read_excel(self.filename, None)
        
        for sheet_name, sheet in excel.items():
            if self.strainer in sheet_name:
                self.sheet = sheet
                return None
                
    def __add_band_index(self):
        max_Gray = self.sheet['Gray'].max()
        band_index = 1
        band_index_list = list()
        
        for idx, row in self.sheet.iterrows():
            if idx > 0 and row['Gray'] == max_Gray and self.sheet.loc[idx-1, 'Gray'] != max_Gray:
                band_index += 1
            band_index_list.append(band_index)
            
        self.sheet['band_index'] = band_index_list
        
    def __add_Lmax(self):
        max_Gray = self.sheet['Gray'].max()
        Lmax_list = list()
        Lmax = self.sheet['Lv theory'][0]
        
        for idx, row in self.sheet.iterrows():
            if row['Gray'] == max_Gray:
                Lmax = row['Lv theory']
            Lmax_list.append(Lmax)
            
        self.sheet['Lmax'] = Lmax_list
        
    def __add_rgb_Dec(self):
        self.sheet['r_Dec'] = self.sheet['Reg_r'].apply(lambda ele: int(str(ele), 16))
        self.sheet['g_Dec'] = self.sheet['Reg_g'].apply(lambda ele: int(str(ele), 16))
        self.sheet['b_Dec'] = self.sheet['Reg_b'].apply(lambda ele: int(str(ele), 16))
        
    def __add_Lv_diff(self):
        self.sheet['Lv_diff'] = self.sheet['Lv'] - self.sheet['Lv theory']
        
    def __get_status(self):    ### 可改进
        self.sheet_init = pandas.DataFrame(columns=self.sheet.columns)
        self.sheet_stop = pandas.DataFrame(columns=self.sheet.columns)
        
        for idx, row in self.sheet.iterrows():
            if idx == 0:
                self.sheet_init = self.sheet_init.append(row, ignore_index=True)
            else:
                if row['Gray'] != self.sheet.loc[idx-1, 'Gray']:
                    init_exist = (self.sheet_init['band_index'] == row['band_index']) & (self.sheet_init['Gray'] == row['Gray'])
                    if True not in init_exist.values:
                        self.sheet_init = self.sheet_init.append(row, ignore_index=True)
                    
                    stop_tab, stop_exist = self.__is_stop_status_last(row)
                    if stop_tab:
                        self.sheet_stop = self.sheet_stop.append(row, ignore_index=True)
                    else:
                        last_idx = self.sheet_stop.loc[stop_exist].index[0]
                        self.sheet_stop.loc[last_idx, :] = row
                    
                    stop_tab, stop_exist = self.__is_stop_status_last(self.sheet.loc[idx-1, :])
                    if stop_tab:
                        self.sheet_stop = self.sheet_stop.append(self.sheet.loc[idx-1, :], ignore_index=True)
                    else:
                        last_idx = self.sheet_stop.loc[stop_exist].index[0]
                        self.sheet_stop.loc[last_idx, :] = self.sheet.loc[idx-1, :]
                else:
                    if idx == self.sheet.shape[0] - 1:
                        stop_tab, stop_exist = self.__is_stop_status_last(row)
                        # print(stop_tab, stop_exist)
                        if stop_tab:
                            self.sheet_stop = self.sheet_stop.append(row, ignore_index=True)
                        else:
                            last_idx = self.sheet_stop.loc[stop_exist].index[0]
                            self.sheet_stop.loc[last_idx, :] = row
                            
        
                        
    def __is_init_status_curr(self, row):
        init_exist = (self.sheet_init['band_index'] == row['band_index']) & (self.sheet_init['Gray'] == row['Gray'])
        if True not in init_exist.values:
            return True
        return False
        
    def __is_stop_status_last(self, row):
        stop_exist = (self.sheet_stop['band_index'] == row['band_index']) & (self.sheet_stop['Gray'] == row['Gray'])
        if True not in stop_exist.values:
            return (True, stop_exist)
        return (False, stop_exist)
                        
                    
# 提取指定目录下的所有excel信息(Gray, Lmax), (r_Dec, g_Dec, b_Dec)
class DataPreprocessing(object):
    
    def __init__(self, path_ref, path_tar):
        self.path_ref = path_ref
        self.path_tar = path_tar
        self.__get_all_filenames()
        
    def __get_all_filenames(self):
        self.filenames_ref = list(os.path.join(self.path_ref, filename) for filename in os.listdir(self.path_ref) if filename.endswith('.xls'))
        self.number_ref = len(self.filenames_ref)
        self.filenames_tar = list(os.path.join(self.path_tar, filename) for filename in os.listdir(self.path_tar) if filename.endswith('.xls'))
        self.number_tar = len(self.filenames_tar)
    
    # 加载所有参考屏信息
    def load_all_ref(self):
        self.df_ref_ins_list = list(ExcelDealing(filename) for filename in self.filenames_ref)
        self.__df_ref_status_list = list(ins.get_status() for ins in self.df_ref_ins_list)
        self.__points_ref_2dlist_init_stop = list()
        
        for df_ref_status in self.__df_ref_status_list:
            self.__points_ref_2dlist_init_stop.append(list(self.__sort_by_Lmax_Gray(sheet_status) for sheet_status in df_ref_status))
    
    # 加载指定目标屏信息
    def load_assigned_tar(self, idx_tar=0):
        self.df_tar_ins = ExcelDealing(self.filenames_tar[idx_tar])
        self.__df_tar_status = self.df_tar_ins.get_status()
        self.__points_tar_2dlist_init_stop = list(self.__sort_by_Lmax_Gray(sheet_status) for sheet_status in self.__df_tar_status)

    def __sort_by_Lmax_Gray(self, sheet_status):
        bands_sorted_by_Lmax = self.__sort_by_Lmax_bef(sheet_status)
        points_2dlist = list()
        for band in bands_sorted_by_Lmax:
            points = self.__sort_by_Gray_aft(band)
            points_2dlist.append(points)
        return points_2dlist
        
    def __sort_by_Lmax_bef(self, sheet_status):
        bands_sorted_by_Lmax = list(sub_df.reset_index(drop=True) for band_index, sub_df in sheet_status.groupby('band_index'))
        bands_sorted_by_Lmax.sort(key=lambda item: item['Lmax'].values[0])
        return bands_sorted_by_Lmax
        
    def __sort_by_Gray_aft(self, sheet_band):
        points_sorted_by_Gray = list(row for idx, row in sheet_band.iterrows())
        points_sorted_by_Gray.sort(key=lambda item: item['Gray'])
        return points_sorted_by_Gray
        
    # 根据索引从目标屏获取D矢量: 初始状态/终止状态
    def get_D_from_tar(self, Lmax_idx, Gray_idx, status='stop'):
        if status == 'stop':
            tab = 1
        else:
            tab = 0
        
        point = self.__points_tar_2dlist_init_stop[tab][Lmax_idx][Gray_idx]
        return numpy.array([point['r_Dec'], point['g_Dec'], point['b_Dec']]).reshape(-1, 1)
        
    # 根据索引从参考屏获取A矩阵: 初始状态/终止状态
    def get_A_from_ref(self, Lmax_idx, Gray_idx, status='stop'):
        if status == 'stop':
            tab = 1
        else:
            tab = 0
        
        points = list(item[tab][Lmax_idx][Gray_idx] for item in self.__points_ref_2dlist_init_stop)
        return numpy.array(list([point['r_Dec'], point['g_Dec'], point['b_Dec']] for point in points)).T
        
        
        
    
                
                
if __name__ == '__main__':
    # path_ref = './reference_screens'
    # path_tar = './target_screens'
    
    # ins = DataPreprocessing(path_ref, path_tar)
    # ins.load_all_ref()
    # ins.load_assigned_tar(0)
    
    # A = ins.get_A_from_ref(0, 0, 'stop')
    # print(A)
    # D = ins.get_D_from_tar(0, 0, 'stop')
    # print(D)
    # print('*'*20)
    # A = ins.get_A_from_ref(0, 1, 'stop')
    # print(A)
    # D = ins.get_D_from_tar(0, 1, 'stop')
    # print(D)
    # print('*'*20)
    # A = ins.get_A_from_ref(0, 2, 'stop')
    # print(A)
    # D = ins.get_D_from_tar(0, 2, 'stop')
    # print(D)
    # print('*'*20)
    # A = ins.get_A_from_ref(0, 3, 'stop')
    # print(A)
    # D = ins.get_D_from_tar(0, 3, 'stop')
    # print(D)
    # print('*'*20)
    # A = ins.get_A_from_ref(0, 4, 'stop')
    # print(A)
    # D = ins.get_D_from_tar(0, 4, 'stop')
    # print(D)

    # filename = './reference_screens/Rack1-Pg1-Link1-JC--20190201011840-OK.xls'
    # filename = './reference_screens/Rack1-Pg2-Link1-JC--20190202084403-OK.xls'
    # strainer = 'W-'
    # excel = ExcelDealing(filename, strainer)
    # sheet_init, sheet_stop = excel.get_status()
    # print(sheet_init.tail(20))
    # print(sheet_stop.tail(20))
    
    #####################################
    path = './ref_new'
    filenames = list(os.path.join(path, filename) for filename in os.listdir(path))
    
    for filename in filenames:
        print(filename)
        try:
            ins = ExcelDealing(filename)
            sheet_init, sheet_stop = ins.get_status()
        except Exception:
            print('异常删除: {}'.format(filename))
            os.remove(filename)
        else:
            if sheet_init.shape != (198, 24) or sheet_init.shape != sheet_stop.shape:
                print('shape不匹配删除: {}'.format(filename))
                os.remove(filename)
        
    


