# 1. 寄存器值Reg与电压值Vol的转换
# 2. 色坐标xy与uv的转换
# 3. A矩阵的计算

import os

import numpy
import pandas


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
        
        
class FileParser(object):
    
    def __init__(self, path):
        self.path = path
        
    def load_files(self):
        self.filename_list_ori = os.listdir(self.path)
        self.filename_list_tra = self.__sorted_filenames(self.filename_list_ori)
        
        self.csv_list = list(pandas.read_csv(os.path.join(self.path, filename), names=['x', 'y', 'Lv', 'Reg_r', 'Reg_g', 'Reg_b']) for filename in self.filename_list_tra)
        self.__add_mean()
        
    def __add_mean(self):
        for i in range(len(self.csv_list)):
            self.csv_list[i] = self.csv_list[i].append(self.csv_list[i].mean(), ignore_index=True)
            
        
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
        self.G_VREG1 = kwargs['G_VREG1'] if 'G_VREG1' in kwargs else 6.5
        self.G_VREF1 = kwargs['G_VREF1'] if 'G_VREF1' in kwargs else 1
        self.G_VT_REG_V = kwargs['G_VT_REG_V'] if 'G_VT_REG_V' in kwargs else 0x12
        self.GrayV0_list = kwargs['GrayV0_list'] if 'GrayV0_list' in kwargs else [6.5] * 17
        
    def __make_trans(self):
        if not hasattr(self, 'G_VREG1'):
            self.feed_params()
            
        trans = FileParser(self.path)
        trans.load_files()
        
        trans.convert2norm(GrayV0_list=self.GrayV0_list, G_VREG1=self.G_VREG1, G_VREF1=self.G_VREF1, G_VT_REG_V=self.G_VT_REG_V)
        
        self.filename_list = trans.filename_list_tra
        self.csv_list = trans.csv_list_new
        self.Gray_list = trans.Gray_list
        
    def get_AMatrix(self, A_saved=True, A_filename=None, epsilon=1.e-9):
        '''
        计算17条band上所有绑点的A矩阵, 缺失则填入前一条band上对应灰阶的A矩阵
        '''
        if not hasattr(self, 'filename_list'):
            self.__make_trans()
            
        self.AMatrix_list = list([None] * len(self.Gray_list) for i in range(17))
            
        for filename, csv in zip(self.filename_list, self.csv_list):
            band_idx, Gray_idx = self.__get_loc(filename)
            AMatrix = self.__calc_AMatrix(csv, epsilon)
            self.AMatrix_list[band_idx][Gray_idx] = AMatrix
            
        self.__fill_AMatrix()                     # 填充缺失矩阵
            
        if A_saved:
            if A_filename is None: A_filename = './Gamma_slope_from_tuning.lua'
            
            self.__save_AMatrix(A_filename)
    
            
    def __save_AMatrix(self, filename):
        with open(filename, 'wt') as f:
            f.write('G_A = {}\n\n')
        
            idx = 1
            for band_idx in range(17):
                for Gray_idx in range(len(self.Gray_list)):
                    AMatrix = self.AMatrix_list[band_idx][Gray_idx]
                    line1 = '-- band{}, W{}\nG_A[{}] = {{\n'.format(band_idx + 1, self.Gray_list[Gray_idx], idx)
                    f.write(line1)
                    
                    line2_list = list()
                    for row in AMatrix.tolist():
                        item = '  {' + ', '.join(str(round(ele, 5)) for ele in row) + '}'
                        line2_list.append(item)
                    line2 = ',\n'.join(line2_list)
                    f.write(line2)
                        
                    line3 = '\n}\n'
                    f.write(line3)
                        
                    idx += 1
                        
        
    def __fill_AMatrix(self):
        for band_idx in range(17):
            for Gray_idx in range(len(self.Gray_list)):
                AMatrix = self.AMatrix_list[band_idx][Gray_idx]
                
                if AMatrix is None:
                    band_seq = self.__get_band_seq(band_idx)
                    for band_idx_sub in band_seq:
                        AMatrix_tmp = self.AMatrix_list[band_idx_sub][Gray_idx]
                        
                        if AMatrix_tmp is not None:
                            self.AMatrix_list[band_idx][Gray_idx] = AMatrix_tmp
                            break
                    else:
                        raise Exception('>>> something confusing happend on (band_idx, Gray_idx) = ({}, {}) <<<'.format(band_idx, Gray_idx))
                        
                    
    def __get_band_seq(self, band_idx):
        seq_bef = list(reversed(range(band_idx)))
        seq_aft = list(range(band_idx + 1, 17))
        
        return seq_bef + seq_aft
        
            
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
        

    def __calc_W(self, A, D, epsilon):
        item1 = numpy.matmul(A.T, A)
        item2 = numpy.identity(A.shape[1]) * epsilon
        item3 = numpy.matmul(A.T, D)
        
        W = numpy.matmul(numpy.linalg.inv(item2 + item1), item3)
        
        return W
        
            
    def __get_loc(self, filename):
        band_num = int(filename[filename.find('band') + 4:filename.find('_')])
        Gray_num = int(filename[filename.find('W') + 1:filename.find('.')])
        
        return band_num - 1, self.Gray_list.index(Gray_num)

        
        
if __name__ == '__main__':
    obj = CalcAMatrix('./540S')
    obj.get_AMatrix()
    
        
        








