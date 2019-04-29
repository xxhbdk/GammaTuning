# 基础解析模块
# 改进方向: 一套统一的表层接口, 再由指定的解析器具体分发

import os
import numpy

from parse.parser_jc import JCEXCEL


class DataPre(object):
    
    def __init__(self, parser):
        self.parser = eval(parser)
        
    def load_all_files(self, tar_path, ref_path):
        self.tar_path = tar_path
        self.ref_path = ref_path
        
        self.__get_filenames()
        self.__get_instances()
        
    def __get_filenames(self):
        ref_files = list(filename for filename in os.listdir(self.ref_path))
        tar_files = list(filename for filename in os.listdir(self.tar_path))
        
        self.files = {self.ref_path: ref_files, self.tar_path: tar_files}
        
    def __get_instances(self):
        ref_instances = list(self.parser(os.path.join(self.ref_path, filename)) for filename in self.files[self.ref_path])
        tar_instances = list(self.parser(os.path.join(self.tar_path, filename)) for filename in self.files[self.tar_path])
        
        self.instances = {self.ref_path: ref_instances, self.tar_path: tar_instances}
        
    def filter_by_shape(self, shape_save=None, remove=False, printable=False):
        '''
        以指定的shape过滤ref_path、tar_path下的excel文件
        shape: (bands_num, Grays_num)
        remove: 是否直接删除不合要求文件
        '''
        if not hasattr(self, 'instances'):
            raise Exception('>>> filter before load <<<')
        if not hasattr(self, '__shapes'):
            self.__get_shapes()
        
        sorted_shapes = sorted(list(self.__shapes.items()), key=lambda item: len(item[1]))
        self.shape_save = sorted_shapes[-1][0] if shape_save is None else tuple(shape_save)
        
        tmp_files = {self.ref_path: list(), self.tar_path: list()}
        tmp_instances = {self.ref_path: list(), self.tar_path: list()}
        
        for shape_curr, files_loc in sorted_shapes:
            if shape_curr != self.shape_save:
                for path, idx in files_loc:
                    if remove:
                        if printable:
                            print('remove: {} --- {}'.format(self.files[path][idx], shape_curr))
                        os.remove(os.path.join(path, self.files[path][idx]))
                    else:
                        if printable:
                            print('noNeed: {} --- {}'.format(self.files[path][idx], shape_curr))
                    tmp_files[path].append(self.files[path][idx])
                    tmp_instances[path].append(self.instances[path][idx])
            else:
                if printable:
                    for path, idx in files_loc:
                        print('remain: {} --- {}'.format(self.files[path][idx], shape_curr))
                    
        for path, files in tmp_files.items():
            for file in files:
                self.files[path].remove(file)
                
        for path, instances in tmp_instances.items():
            for instance in instances:
                self.instances[path].remove(instance)
        
    def __get_shapes(self):
        self.__shapes = dict()
        
        for path, instances in self.instances.items():
            for idx, ins in enumerate(instances):
                
                shape = ins.shape
                
                if shape not in self.__shapes:
                    self.__shapes[shape] = [(path, idx)]
                else:
                    self.__shapes[shape].append((path, idx))
                    
    def get_D_and_A(self, tar_idx=0, ref_idx_list=None):
        '''
        获取所有绑点的D矢量与A矩阵
        index_tar: 目标屏编号
        index_ref_list: 参考屏编号列表
        return: {(Gray, Lmax): (D, A)}
        '''
        if not hasattr(self, 'shape_save'):
            self.filter_by_shape()
        if ref_idx_list is None:
            ref_idx_list = range(len(self.instances[self.ref_path]))
        
        self.__get_D_and_A(tar_idx, ref_idx_list)
            
        return self.init_D_and_A, self.stop_D_and_A
        
    def __get_D_and_A(self, tar_idx, ref_idx_list):
        self.init_D_and_A = dict()
        self.stop_D_and_A = dict()
        
        self.tar_ins = self.instances[self.tar_path][tar_idx]                                           # 当前目标屏实例 -> 索引顺序参考
        self.ref_ins_list = list(self.instances[self.ref_path][ref_idx] for ref_idx in ref_idx_list)    # 当前参考屏实例列表
        
        for GrayLmax, stop_rgb in self.tar_ins.get_attrs()['stop_GrayLmax2rgb'].items():
            
            D_init = numpy.array(self.tar_ins.get_attrs()['init_GrayLmax2rgb'][GrayLmax]).reshape(-1, 1)
            D_stop = numpy.array(stop_rgb).reshape(-1, 1)
            
            A_init, A_stop = self.__extract_A(GrayLmax)
            
            self.init_D_and_A[GrayLmax] = (D_init, A_init)
            self.stop_D_and_A[GrayLmax] = (D_stop, A_stop)
            
    def __extract_A(self, GrayLmax):
        A_init, A_stop = list(), list()
        
        for ref_ins in self.ref_ins_list:
            A_init.append(ref_ins.get_attrs()['init_GrayLmax2rgb'][GrayLmax])
            A_stop.append(ref_ins.get_attrs()['stop_GrayLmax2rgb'][GrayLmax])
            
        return numpy.array(A_init).T, numpy.array(A_stop).T
        
    
    
if __name__ == '__main__':
    obj = DataPre('JCEXCEL')
    obj.load_all_files('../screens_tar', '../screens_ref')
    # print(obj.instances)
    obj.filter_by_shape()
    # print(obj.instances)
    # init, stop = obj.get_D_and_A()
    # print(stop)