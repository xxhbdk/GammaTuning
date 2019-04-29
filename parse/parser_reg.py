# 基础解析器3: 解析InitReg.csv

import pandas

class StateSheet(object):
    
    def __init__(self, state_sheet):
        self.state_sheet = state_sheet
        
    @property
    def shape(self):
        if not hasattr(self, '__bands_num'):
            self.__get_shape()
            
        return self.__bands_num, self.__Grays_num
        
    def __get_shape(self):
        self.__bands_num = self.state_sheet['band_index'].value_counts().count()
        self.__Grays_num = self.state_sheet['Gray'].value_counts().count()
        
    def get_attrs(self):
        '''
        返回一个大字典, 主要包含:
        bands
        band2Gray -> Gray顺序校验
        Grayband2rgb
        '''
        if not hasattr(self, 'attrs'):
            self.__get_attrs()
            
        return self.attrs
        
    def __get_attrs(self):
        self.attrs = dict()
        
        bands = self.state_sheet['band_index'].value_counts().index.sort_values().to_list()
        band2Gray = dict()
        Grayband2rgb = dict()
        
        for idx, row in self.state_sheet.iterrows():
            band_index, Gray, r_Dec, g_Dec, b_Dec = row[['band_index', 'Gray', 'r_Dec', 'g_Dec', 'b_Dec']]
            
            if band_index not in band2Gray:
                band2Gray[band_index] = [Gray]
            else:
                band2Gray[band_index].append(Gray)
                
            Grayband2rgb[(Gray, band_index)] = [r_Dec, g_Dec, b_Dec]
            
        self.attrs['bands'] = bands
        
        # 将每条band上的Gray由小到大进行排序
        for val in band2Gray.values():
            val.sort()
        self.attrs['band2Gray'] = band2Gray
        
        self.attrs['Grayband2rgb'] = Grayband2rgb
        
    def __repr__(self):
        return 'State<bands_num: {}, Grays_num: {}>@{}'.format(*self.shape, id(self))
        


class REGCSV(object):
    
    def __init__(self, filename):
        self.filename = filename
        
    def get_sheet(self):
        if not hasattr(self, 'sheet'):
            self.__get_sheet()
            self.__add_cols()
            self.__del_cols()
            
        return self.sheet
        
    def __get_sheet(self):
        try:
            self.sheet = pandas.read_csv(self.filename)
        except Exception:
            exit('>>> something wrong with "{}" <<<'.format(self.filename))
            
    def __add_cols(self):
        if len(self.sheet.columns) == 4:
            self.sheet.columns = ['index', 'Reg_r', 'Reg_g', 'Reg_b']
        else:
            self.sheet.reset_index(inplace=True)
            self.sheet.columns = ['index', 'Reg_r', 'Reg_g', 'Reg_b'] + list('tab{}'.format(i) for i in range(len(self.sheet.columns) - 4))
        
        self.sheet['screen_index'] = self.sheet['index'].apply(lambda ele: ele.split('-')[0]).astype(int)
        self.sheet['band_index'] = self.sheet['index'].apply(lambda ele: ele.split('-')[1]).astype(int)
        self.sheet['Gray'] = self.sheet['index'].apply(lambda ele: ele.split('-')[-1]).astype(int)
        
        self.sheet['r_Dec'] = self.sheet['Reg_r'].apply(lambda ele: int(str(ele), 16))
        self.sheet['g_Dec'] = self.sheet['Reg_g'].apply(lambda ele: int(str(ele), 16))
        self.sheet['b_Dec'] = self.sheet['Reg_b'].apply(lambda ele: int(str(ele), 16))
        
    def __del_cols(self):
        labels = ['index', 'Reg_r', 'Reg_g', 'Reg_b'] + list('tab{}'.format(i) for i in range(len(self.sheet.columns) - 10))
        self.sheet.drop(labels=labels, axis=1, inplace=True)
        
    @property
    def shape(self):
        '''
        返回band数与Gray数(假设所有screen上的所有band的Gray数目相同)
        '''
        if not hasattr(self, 'sheet'):
            self.get_sheet()
        if not hasattr(self, '__bands_num'):
            self.__get_shape()
        
        return self.__screens_num, self.__bands_num, self.__Grays_num
    
    def __get_shape(self):
        self.__screens_num = self.sheet['screen_index'].value_counts().count()
        self.__bands_num = self.sheet['band_index'].value_counts().count()
        self.__Grays_num = self.sheet['Gray'].value_counts().count()
        
    def get_states(self):
        '''
        提取状态表格, 构建状态对象字典 {screen_index: state_sheet}
        '''
        if not hasattr(self, 'sheet'):
            self.get_sheet()
        if not hasattr(self, 'sheet_dict'):
            self.__get_states()
            
        return self.sheet_dict
        
    def __get_states(self):
        self.sheet_dict = dict()
        
        for screen_index, sheet_sub1 in self.sheet.groupby('screen_index'):
            obj = StateSheet(sheet_sub1)
            self.sheet_dict[screen_index] = obj
            
    def __repr__(self):
        return 'Reg<screens_num: {}, bands_num: {}, Grays_num: {}>@{}'.format(*self.shape, id(self))
            

            
if __name__ == '__main__':
    filename = 'CH1_InitReg.csv'
    # filename = 'initVal.csv'
    obj = REGCSV(filename)
    obj.get_states()
    for state in obj.sheet_dict.values():
        print(state.get_attrs()['bands'])
    print(obj)
    # obj.get_sheet()
    # print(obj.shape)
    # print(obj.get_states())


