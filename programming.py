# 二次规划求解模块

import numpy

from preprocessing import DataPreprocessing

class Solver(object):
    
    # 采用增广拉格朗日法求解该二次规划 --> 当前精度下大约1ms完成求解过程, 迭代4次
    def solve2norm(self, A, D, epsilon=1.e-7):
        rows, cols = A.shape
        c = 1                                  # 可调参数初始化
        beta = numpy.ones(shape=(3, 1))        # 对偶变量初始化
        w = numpy.matmul(numpy.linalg.inv(numpy.identity(cols) + c * numpy.matmul(A.T, A)), c * numpy.matmul(A.T, D) - numpy.matmul(A.T, beta))
        beta_delta = c * (numpy.matmul(A, w) - D)
        
        while numpy.linalg.norm(beta_delta) > epsilon:
            beta += beta_delta
            w = numpy.matmul(numpy.linalg.inv(numpy.identity(cols) + c * numpy.matmul(A.T, A)), c * numpy.matmul(A.T, D) - numpy.matmul(A.T, beta))
            beta_delta = c * (numpy.matmul(A, w) - D)
            print(numpy.linalg.norm(beta_delta))   #################

        return w
        
    def solve1norm(self, A, D, epsilon=1.e-8):
        pass
        

##############################################################################################
class QuadraticPro(object):
    
    def __init__(self, path_ref, path_tar):
        self.path_ref = path_ref
        self.path_tar = path_tar
        
    def load_all_ref(self):
        self.data_ins = DataPreprocessing(self.path_ref, self.path_tar)
        self.data_ins.load_all_ref()
        
    def load_assigned_tar(self, tar_idx=0):
        self.data_ins.load_assigned_tar(tar_idx)
        
    def load_data(self, tar_idx=0):
        '''
        tar_index: 目标屏编号
        '''
        self.load_all_ref()
        self.load_assigned_tar(tar_idx)
        
    def get_A_D(self, Lmax_idx=0, Gray_idx=0, status='stop'):
        A = self.data_ins.get_A_from_ref(Lmax_idx, Gray_idx, status)
        D = self.data_ins.get_D_from_tar(Lmax_idx, Gray_idx, status)
        return (A, D)
    
    # 采用增广拉格朗日法求解该二次规划 --> 当前精度下大约1ms完成求解过程, 迭代4次
    def solve2norm(self, A, D, epsilon=1.e-8):
        rows, cols = A.shape
        c = 1                                  # 可调参数初始化
        beta = numpy.ones(shape=(3, 1))        # 对偶变量初始化
        w = numpy.matmul(numpy.linalg.inv(numpy.identity(cols) + c * numpy.matmul(A.T, A)), c * numpy.matmul(A.T, D) - numpy.matmul(A.T, beta))
        beta_delta = c * (numpy.matmul(A, w) - D)
        
        while numpy.linalg.norm(beta_delta) > epsilon:
            beta += beta_delta
            w = numpy.matmul(numpy.linalg.inv(numpy.identity(cols) + c * numpy.matmul(A.T, A)), c * numpy.matmul(A.T, D) - numpy.matmul(A.T, beta))
            beta_delta = c * (numpy.matmul(A, w) - D)

        return w
        
    def solve1norm(self, A, D, epsilon=1.e-8):
        pass
        
        
        
if __name__ == '__main__':
    path_ref = './reference_screens'
    path_tar = './target_screens'
    ins = QuadraticPro(path_ref, path_tar)
    ins.load_data(tar_idx=1)
    #################################
    # try:
        # Lmax_idx = 0
        # Gray_idx = 1
        # while True:
            # A, D = ins.get_A_D(Lmax_idx, Gray_idx)
            # w = ins.solve2norm(A, D, 1.e-8)
            # Lmax_idx += 1
            # print(w[0, 0])
    # except Exception as e:
        # print('Lmax={}, Gray={}'.format(Lmax_idx, Gray_idx))
        # print(e)
    #################################
    # A, D = ins.get_A_D(0, 0)
    # w = ins.solve2norm(A, D, 1.e-8)
    # A1, D1 = ins.get_A_D(0, 1)
    # w1 = ins.solve2norm(A, D, 1.e-8)
    # print(numpy.matmul(A1, w))
    # print(D1)
    # print('*'*20)
    # A2, D2 = ins.get_A_D(0, 1, 'init')
    # print(D2)
    import time
    time1 = time.time()
    try:
        Lmax_idx, Gray_idx = 1, 32
        A_init, D_init = ins.get_A_D(Lmax_idx, Gray_idx, 'init')
        w= numpy.array([1 / A_init.shape[1]] * A_init.shape[1]).reshape(-1, 1)
        
        while Gray_idx >= 0:
            A_init, D_init = ins.get_A_D(Lmax_idx, Gray_idx, 'init')
            A_stop, D_stop = ins.get_A_D(Lmax_idx, Gray_idx, 'stop')
            # print('均匀加权:')
            # print(numpy.around(A_stop.mean(axis=1)))
            # print('非均匀加权:')
            # print(numpy.around(numpy.matmul(A_stop, w)).reshape(-1))
            # print('原始初值:')
            # print(D_init.reshape(-1))
            # print('迭代终值:')
            # print(D_stop.reshape(-1))
            w = ins.solve2norm(A_stop, D_stop)
            print(w)
            Gray_idx -= 1
            print('*'*20)
            
    except Exception as e:
        print(Lmax_idx, Gray_idx)
        print(e)
        
    print('耗时: ', time.time() - time1)
        
        