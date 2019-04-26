# 基础解析器2: 解析Link.xls

import numpy
import pandas

class LinkExcel(object):
    
    def __init__(self, filename):
        self.filename = filename
        
    def get_sheet(self, drop_labels=['Theory Lv', 'LvOKNG', 'Regulate x', 'Theory x', 'xMax', 'xMin', 'xOKNG', 'Regulate y', 'Theory y', 'yMax', 'yMin', 'yOKNG']):
        if not hasattr(self, 'sheet'):
            self.__get_sheet()
            self.__add_cols()
            self.__del_cols(drop_labels)
        
        return self.sheet
        
    def __get_sheet(self):
        try:
            excel = pandas.read_excel(self.filename, None)
            
            for sheet_name, sheet in excel.items():
                if 'Tuning Data' in sheet_name:
                    self.sheet = sheet
                    return None
        except Exception:
            exit('>>> something wrong with "{}" <<<'.format(self.filename))
            
    def __add_cols(self):
        self.sheet['Band number'] = self.sheet['Band number'].apply(lambda ele: ele[ele.find('-') + 1:]).astype(int)
        self.sheet['Gray'] = self.sheet['Gray'].apply(lambda ele: ele[ele.find('-') + 1:]).astype(int)
        self.sheet['Regulate time'] = self.sheet['Regulate time'].apply(lambda ele: ele[:-2]).astype(int)
        self.sheet['Lv position'] = (self.sheet['Actual Lv'] - self.sheet['Theory Min Lv']) / (self.sheet['Theory Max Lv'] - self.sheet['Theory Min Lv'])  
        self.sheet['x position'] = (self.sheet['Regulate x'] - self.sheet['xMin']) / (self.sheet['xMax'] - self.sheet['xMin'])
        self.sheet['y position'] = (self.sheet['Regulate y'] - self.sheet['yMin']) / (self.sheet['yMax'] - self.sheet['yMin'])
    
    def __del_cols(self, labels=None):
        self.sheet.drop(labels=labels, axis=1, inplace=True)
        
    @property
    def shape(self):
        if not hasattr(self, 'sheet'):
            self.get_sheet()
        if not hasattr(self, '__bands_num'):
            self.__get_shape()
            
        return self.__bands_num, self.__Grays_num
            
    def __get_shape(self):
        self.__bands_num = self.sheet['Band number'].value_counts().count()
        self.__Grays_num = self.sheet['Band number'].value_counts().iloc[0]
    
    
    def get_attrs(self):
        '''
        返回一个大字典, 主要包含:
        bands
        band2Lmax
        band2sheet
        '''
        if not hasattr(self, 'sheet'):
            self.get_sheet()
        if not hasattr(self, 'attrs'):
            self.__get_attrs()
        
        return self.attrs
        
    def __get_attrs(self):
        self.attrs = dict()
        
        bands = self.sheet['Band number'].value_counts().index.sort_values().to_list()
        
        max_Gray = self.sheet['Gray'].max()
        band2Lmax = dict()
        band2sheet = dict()
        
        for idx, row in self.sheet[self.sheet['Gray'] == max_Gray].iterrows():
            band2Lmax[row['Band number']] = row['Theory Max Lv']
            
        for Band_number, sheet_sub in self.sheet.groupby('Band number'):
            sheet_new = sheet_sub.sort_values(by='Gray')
            sheet_new.reset_index(drop=True, inplace=True)
            
            band2sheet[Band_number] = sheet_new
        
        self.attrs['bands'] = bands
        self.attrs['band2Lmax'] = band2Lmax
        self.attrs['band2sheet'] = band2sheet
    
    def __repr__(self):
        return 'Link<bands_num: {}, Grays_num: {}>@{}'.format(*self.shape, id(self))
    
    
if __name__ == '__main__':
    filename = 'Rack1-Pg1-Link1--20190416194931-OK.xls'
    obj = LinkExcel(filename)
    
    attrs = obj.get_attrs()
    print(obj.attrs)
    # print(attrs['band2sheet'][list(attrs['band2Lmax'].keys())[0]])
    
    # obj.get_sheet()
    # print(obj.sheet)
    
    