# 数据处理模块(针对第一批Gamma中"Link-JC--"数据)

# 提取初始状态与终止状态
# 提取特征向量
# 获取band_num与gray_num

# 提取A矩阵及特征矩阵FA(参考屏) -> 指定编号列表
# 提取D矢量及特征矢量FD(目标屏) -> 指定编号

import os
import numpy
import pandas


class ExcelDealing(object):
    
    def __init__(self, filename):
        self.filename = filename
    
    def get_sheet(self):
        if not hasattr(self, 'sheet'):
            self.__get_sheet()
            self.__add_band_index()
            self.__add_Lmax()
            self.__add_rgb_Dec()
        
        return self.sheet
        
    def get_status(self):
        if not hasattr(self, 'sheet'):
            self.get_sheet()
        if not hasattr(self, 'sheet_init'):
            self.__get_status()
        
        return self.sheet_init, self.sheet_stop
        
    def get_feature_vector(self):
        '''
        当前特征向量为Gray=255时的初始Lv(固定rgb寄存器值)
        '''
        if not hasattr(self, 'sheet_init'):
            self.get_status()
        if not hasattr(self, 'feature_vector'):
            self.__get_feature_vector()
        
        return self.feature_vector
        
    def get_rgb(self):
        '''
        返回{(Gray, Lmax): (r, g, b)}字典
        辅助校验参考屏
        '''
        if not hasattr(self, 'sheet_init'):
            self.get_status()
        if not hasattr(self, 'rgb_init'):
            self.__get_rgb()
        
        return self.rgb_init, self.rgb_stop
        
    def get_sorted_GrayLmaxdict(self):
        '''
        排序策略, 影响最终的计算顺序
        current: Gray由高向低
        '''
        if not hasattr(self, 'sheet_init'):
            self.get_status()
        if not hasattr(self, 'GrayLmaxdict'):
            self.__get_sorted_GrayLmaxdict()
            
        return self.GrayLmaxdict
        
        
    @property
    def shape(self):
        '''
        返回band数与Gray数(假设所有band上的Gray数目相同)
        '''
        if not hasattr(self, 'sheet_init'):
            self.get_status()
        if not hasattr(self, 'bands_num'):
            self.__get_shape()
        
        return self.__bands_num, self.__Grays_num
        
    def __get_sheet(self):
        excel = pandas.read_excel(self.filename, None)
        
        for sheet_name, sheet in excel.items():
            if 'W-' in sheet_name:
                self.sheet = sheet
                return None
                
        raise Exception('>>> Lack of assigned sheet with "W-" in "{}" <<<'.format(self.filename))
        
    def __add_band_index(self):
        max_Gray = self.sheet['Gray'].max()
        
        band_index = 0
        band_index_list = list()
        
        for idx, row in self.sheet.iterrows():
            if idx > 0 and row['Gray'] == max_Gray and self.sheet.loc[idx-1, 'Gray'] != max_Gray:
                band_index += 1
            band_index_list.append(band_index)
            
        self.sheet['band_index'] = band_index_list
        
    def __add_Lmax(self):
        max_Gray = self.sheet['Gray'].max()
        
        Lmax = self.sheet['Lv theory'][0]
        Lmax_list = list()
        
        for idx, row in self.sheet.iterrows():
            if row['Gray'] == max_Gray:
                Lmax = row['Lv theory']
            Lmax_list.append(Lmax)
            
        self.sheet['Lmax'] = Lmax_list
        
    def __add_rgb_Dec(self):
        self.sheet['r_Dec'] = self.sheet['Reg_r'].apply(lambda ele: int(str(ele), 16))
        self.sheet['g_Dec'] = self.sheet['Reg_g'].apply(lambda ele: int(str(ele), 16))
        self.sheet['b_Dec'] = self.sheet['Reg_b'].apply(lambda ele: int(str(ele), 16))
        
    def __get_status(self):
        self.sheet_init = pandas.DataFrame(columns=self.sheet.columns)
        self.sheet_stop = pandas.DataFrame(columns=self.sheet.columns)
        
        for band_index, sheet_sub1 in self.sheet.groupby('band_index'):
            for Gray, sheet_sub2 in sheet_sub1.groupby('Gray'):
                self.sheet_init = self.sheet_init.append(sheet_sub2.iloc[0, :], ignore_index=True)
                self.sheet_stop = self.sheet_stop.append(sheet_sub2.iloc[-1, :], ignore_index=True)
                
    def __get_feature_vector(self):
        self.feature_vector = self.sheet_init.loc[self.sheet_init['Gray'] == 255, :]['Lv'].values
        
    def __get_rgb(self):
        self.rgb_init, self.rgb_stop = dict(), dict()
        
        for idx, row in self.sheet_init.iterrows():
            self.rgb_init[(row['Gray'], row['Lmax'])] = (row['r_Dec'], row['g_Dec'], row['b_Dec'])
            
        for idx, row in self.sheet_stop.iterrows():
            self.rgb_stop[(row['Gray'], row['Lmax'])] = (row['r_Dec'], row['g_Dec'], row['b_Dec'])
        
    def __get_sorted_GrayLmaxdict(self):
        self.GrayLmaxdict = dict((band_index, list()) for band_index in range(self.shape[0]))
        for idx, row in self.sheet_init.iterrows():
            self.GrayLmaxdict[row['band_index']].append((row['Gray'], row['Lmax']))
        
        for GrayLmax_band in self.GrayLmaxdict.values():
            GrayLmax_band.sort(key=lambda item: item[0], reverse=True)
        
    def __get_shape(self):
        self.__bands_num = self.sheet_init.loc[self.sheet_init['Gray'] == self.sheet_init['Gray'].max(), :].shape[0]
        self.__Grays_num = self.sheet_init.loc[self.sheet_init['band_index'] == 1, :].shape[0]
        
        
        
        

class DataPreprocessing(object):
    
    def __init__(self, path_ref, path_tar):
        self.path_ref = path_ref
        self.path_tar = path_tar
        
        self.__get_filenames()
        
    def load_all_files(self):
        self.ref_list = list(ExcelDealing(filename) for filename in self.filenames_ref)
        self.tar_list = list(ExcelDealing(filename) for filename in self.filenames_tar)
        
    def get_D_and_A(self, index_tar=0, index_ref_list=None):
        '''
        获取绑点的D矢量与A矩阵, 要求所有绑点协调一致
        index_tar: 目标屏编号
        index_ref_list: 参考屏编号列表
        return: {(Gray, Lmax): (D, A)}
        '''
        if not hasattr(self, 'ref_list'):
            self.load_all_files()
        if index_ref_list is None:
            index_ref_list = range(self.number_ref)
        self.__filter_ref(index_tar, index_ref_list)
        self.__get_D_and_A()                          # 从self.__curr_tar与self.__curr_ref_list中提取信息
        
        return self.D_and_A_init, self.D_and_A_stop
        
    def __get_filenames(self):
        self.filenames_ref = list(os.path.join(self.path_ref, filename) for filename in os.listdir(self.path_ref) if filename.endswith('.xls'))
        self.number_ref = len(self.filenames_ref)
        self.filenames_tar = list(os.path.join(self.path_tar, filename) for filename in os.listdir(self.path_tar) if filename.endswith('.xls'))
        self.number_tar = len(self.filenames_tar)
        
    def __filter_ref(self, index_tar, index_ref_list):
        self.__curr_tar = self.tar_list[index_tar]
        self.__curr_ref_list = list()
        
        for idx, curr_ref in enumerate(self.ref_list):
            if idx in index_ref_list and self.__is_needed(self.__curr_tar, curr_ref):
                self.__curr_ref_list.append(curr_ref)
        
        self.curr_ref_num = len(self.__curr_ref_list)
                
    def __is_needed(self, tar, ref):
        rgb_init_tar, _ = tar.get_rgb()
        rgb_init_ref, _ = ref.get_rgb()
        
        for GrayLmax in rgb_init_tar:
            if GrayLmax not in rgb_init_ref:
                return False
        return True

    def __get_D_and_A(self):
        self.D_and_A_init, self.D_and_A_stop = dict(), dict()
            
        rgb_init_tar, rgb_stop_tar = self.__curr_tar.get_rgb()
        rgb_init_ref_list = list(ref.get_rgb()[0] for ref in self.__curr_ref_list)
        rgb_stop_ref_list = list(ref.get_rgb()[1] for ref in self.__curr_ref_list)
        for GrayLmax in rgb_init_tar:
            D_init = numpy.array(rgb_init_tar[GrayLmax]).reshape(-1, 1)
            A_init = self.__get_assigned_D(GrayLmax, rgb_init_ref_list)
            D_stop = numpy.array(rgb_stop_tar[GrayLmax]).reshape(-1, 1)
            A_stop = self.__get_assigned_D(GrayLmax, rgb_stop_ref_list)
            self.D_and_A_init[GrayLmax] = (D_init, A_init)
            self.D_and_A_stop[GrayLmax] = (D_stop, A_stop)
            
    def __get_assigned_D(self, GrayLmax, rgb_ref_list):
        assigned_D = list()
        
        for rgb_ref in rgb_ref_list:
            assigned_D.append(rgb_ref[GrayLmax])
            
        return numpy.array(assigned_D).T

        
if __name__ == '__main__':
    import time
    from optimize import InertiaSolver
    solver = InertiaSolver()
    path_ref = './screens_ref'
    path_tar = './screens_tar'

    obj = DataPreprocessing(path_ref, path_tar)
    D_and_A_init, D_and_A_stop = obj.get_D_and_A()
    for key, value in D_and_A_stop.items():
        print(key)
        D = value[0]
        A = value[1]
        print(A.shape)
        # try:
            # time1 = time.time()
            # W = solver.solve2norm(A, D)
            # print('Cost Time: {}\n{} == {}\n{}'.format(time.time() - time1, numpy.matmul(A, W), D, W))
        # except Exception as e:
            # print(e)
            # print(A)
            # print(D)
            # numpy.savetxt('./D.txt', D)
            # numpy.savetxt('./A.txt', A)
            # break
    print('#'*120)
    # for key, value in D_and_A_stop.items():
        # print(key)
        # print(value[0])
        # print(value[1])
        # print('*'*30)
    

    
    # import time
    # filename = './screens_ref/Rack1-Pg2-Link1-JC--20190202165050-OK.xls'
    # obj = ExcelDealing(filename)
    # print(time.time())
    # pritn(obj.get_sheet())
    # print(time.time())
    # print(obj.get_sheet())
    # print(time.time())
    # rgb_init, rgb_stop = obj.get_rgb()
    # print(rgb_init)
    # print(obj.shape)
    # sheet = obj.get_sheet()
    # sheet_init, sheet_stop = obj.get_status()
    # feature_vector = obj.get_feature_vector()
    # print(sheet_init)
    # print(sheet_stop)
    # print(feature_vector)
    # print(obj.shape)
                








