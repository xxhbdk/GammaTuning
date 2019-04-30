# 基础解析器1: 解析Link-JC.xls

import numpy
import pandas


class JCEXCEL(object):

    def __init__(self, filename):
        self.filename = filename

    def get_sheet(self, drop_labels=['Reg_r', 'Reg_g', 'Reg_b', 'Z', 'FlagMethod', 'DPTotalStep', 'TraditonTotalStep', 'TotalTimes', 'Rev1', 'Rev2', 'Rev3', 'Rev4']):
        if not hasattr(self, 'sheet'):
            self.__get_sheet()
            self.__add_cols()
            self.__del_cols(drop_labels)
            
        return self.sheet
        
    def __get_sheet(self):
        try:
            excel = pandas.read_excel(self.filename, None)
                
            for sheet_name, sheet in excel.items():
                if 'W-' in sheet_name:
                    self.sheet = sheet
                    return None
        except Exception:
            exit('>>> something wrong with "{}" <<<'.format(self.filename))
        
    def __add_cols(self):
        self.__max_Gray = self.sheet['Gray'].max()
        
        band_index = 0
        band_index_list = list()
        
        Lmax = self.sheet['Lv theory'][0]
        Lmax_list = list()
        
        for idx, row in self.sheet.iterrows():
            if row['Gray'] == self.__max_Gray:
                Lmax = row['Lv theory']
                
                if idx > 0 and self.sheet.loc[idx-1, 'Gray'] != self.__max_Gray:
                    band_index += 1
                
            Lmax_list.append(Lmax)
            band_index_list.append(band_index)
        
        self.sheet['Lmax'] = Lmax_list
        self.sheet['band_index'] = band_index_list
        
        self.sheet['r_Dec'] = self.sheet['Reg_r'].apply(lambda ele: int(str(ele), 16))
        self.sheet['g_Dec'] = self.sheet['Reg_g'].apply(lambda ele: int(str(ele), 16))
        self.sheet['b_Dec'] = self.sheet['Reg_b'].apply(lambda ele: int(str(ele), 16))
        
        self.sheet['Lv precision'] = (self.sheet['Lv'] - self.sheet['Lv theory']).abs() / self.sheet['Lv theory']
    
    def __del_cols(self, labels=None):
        self.sheet.drop(labels=labels, axis=1, inplace=True)
    
    def get_states(self):
        if not hasattr(self, 'sheet'):
            self.get_sheet()
        if not hasattr(self, 'stop_sheet'):
            self.__get_states()

        return self.init_sheet, self.stop_sheet
    
    def __get_states(self):
        self.init_sheet = pandas.DataFrame(columns=self.sheet.columns)
        self.stop_sheet = pandas.DataFrame(columns=self.sheet.columns)
        
        for band_index, sheet_sub1 in self.sheet.groupby('band_index'):
            for Gray, sheet_sub2 in sheet_sub1.groupby('Gray'):
                self.init_sheet = self.init_sheet.append(sheet_sub2.iloc[0, :], ignore_index=True)
                self.stop_sheet = self.stop_sheet.append(sheet_sub2.iloc[-1, :], ignore_index=True)
    
    @property
    def shape(self):
        '''
        返回band数与Gray数(假设所有band上的Gray数目相同)
        '''
        if not hasattr(self, 'stop_sheet'):
            self.get_states()
        if not hasattr(self, '__bands_num'):
            self.__get_shape()
            
        return self.__bands_num, self.__Grays_num
    
    @property
    def precision(self):
        '''
        获取计算精度abs(Lv - Lt) / Lt
        '''
        if not hasattr(self, 'stop_sheet'):
            self.get_states()
        if not hasattr(self, '__precision'):
            self.__get_prec()
            
        return self.__precision
        
    def __get_prec(self):
        self.__precision = self.sheet['Lv precision'].mean()
        
    def __get_shape(self):
        self.__bands_num = self.stop_sheet.loc[self.stop_sheet['Gray'] == self.__max_Gray, :].shape[0]
        self.__Grays_num = self.stop_sheet.loc[self.stop_sheet['band_index'] == 0, :].shape[0]
    
    def get_attrs(self):
        '''
        返回一个大字典, 主要包含:
        bands
        band2Lmax
        band2GrayLmax -> GrayLmax顺序校验
        stop_GrayLmax2rgb
        init_GrayLmax2rgb
        '''
        if not hasattr(self, 'stop_sheet'):
            self.get_states()
        if not hasattr(self, 'attrs'):
            self.__get_attrs()
            
        return self.attrs
        
    def __get_attrs(self):
        self.attrs = dict()
        
        bands = list()
        band2Lmax = dict()
        band2GrayLmax = dict()
        stop_GrayLmax2rgb = dict()
        init_GrayLmax2rgb = dict()
        
        for idx, row in self.stop_sheet.iterrows():
            stop_band_index, stop_Gray, stop_Lmax = row[['band_index', 'Gray', 'Lmax']]

            if stop_band_index not in bands:
                bands.append(stop_band_index)
                band2Lmax[stop_band_index] = stop_Lmax
                
            if stop_band_index not in band2GrayLmax:
                band2GrayLmax[stop_band_index] = [(stop_Gray, stop_Lmax)]
            else:
                band2GrayLmax[stop_band_index].append((stop_Gray, stop_Lmax))
            
            stop_r_Dec, stop_g_Dec, stop_b_Dec = row[['r_Dec', 'g_Dec', 'b_Dec']]
            stop_GrayLmax2rgb[(stop_Gray, stop_Lmax)] = [stop_r_Dec, stop_g_Dec, stop_b_Dec]
            
            init_Gray, init_Lmax, init_r_Dec, init_g_Dec, init_b_Dec = self.init_sheet.loc[idx, ['Gray', 'Lmax', 'r_Dec', 'g_Dec', 'b_Dec']]
            init_GrayLmax2rgb[(init_Gray, init_Lmax)] = [init_r_Dec, init_g_Dec, init_b_Dec]
            
        self.attrs['bands'] = bands
        self.attrs['band2Lmax'] = band2Lmax
        
        # 将每条band上的Gray由小到大进行排序
        for val in band2GrayLmax.values():
            val.sort(key=lambda item: item[0])
        self.attrs['band2GrayLmax'] = band2GrayLmax
        
        # 终止状态下的rgb
        self.attrs['stop_GrayLmax2rgb'] = stop_GrayLmax2rgb
        
        # 初始状态下的rgb
        self.attrs['init_GrayLmax2rgb'] = init_GrayLmax2rgb
        
        
    def get_dominants(self):
        '''
        屏的特征: 所有band的dominant directions
        '''
        if not hasattr(self, 'attrs'):
            self.get_attrs()
        if not hasattr(self, 'stop_dominates'):
            self.__get_dominants()
            
        return self.dominants
        
    def __get_dominants(self):
        self.dominants = dict()
    
        band2GrayLmax = self.attrs['band2GrayLmax']
        stop_GrayLmax2rgb = self.attrs['stop_GrayLmax2rgb']
        
        for band_index, GrayLmax_list in band2GrayLmax.items():
            rgb_arr = numpy.array(list(stop_GrayLmax2rgb[GrayLmax] for GrayLmax in GrayLmax_list))
            rgb_arr -= rgb_arr.mean(axis=0)
            
            dominant = self.__cal_dominant(rgb_arr)
            self.dominants[band_index] = dominant
            
    def __cal_dominant(self, rgb_arr):
        '''
        约束本征矢的长度为1
        '''
        eigvals, eigvecs = numpy.linalg.eig(numpy.matmul(rgb_arr.T, rgb_arr))
        eigvecs = eigvecs / numpy.linalg.norm(eigvecs, axis=0)
        
        max_idx = eigvals.argmax()
        dominant = eigvecs[:, max_idx]
        
        return dominant
        
    def __repr__(self):
        return 'LinkJC<bands_num: {}, Grays_num: {}>@{}'.format(*self.shape, id(self))
        
        
if __name__ == '__main__':

    filename = 'Rack1-Pg1-Link1-JC--20190418175036-OK.xls'
    obj = JCEXCEL(filename)
    obj.get_sheet()
    
    print(obj.precision)
    # import os
    # filenames = list(os.path.join('./Gammaexcel', filename) for filename in os.listdir('./Gammaexcel'))
    
    # for filename in filenames:
        # print(filename)
        # obj = LinkJCExcel(filename)
        # print(obj.shape)
        
    
    
    
    
