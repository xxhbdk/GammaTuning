# 惯性优化器

import numpy


class InertiaSolver(object):
    
    def solve2norm(self, A, D, W_old=None, omega=1, epsilon1=1.e-6, epsilon2=1.e+6):
        '''
        优化问题:
        min Norm(W - W_old, 2)^2 / 2 + omega * Norm(W, 2)^2 / 2
        s.t. AW = D
        where:
        A = numpy.array([[r1, r2, r3, ...], [g1, g2, g3, ...], [b1, b2, b3, ...], ...])
        D = numpy.array([[r], [g], [b]])
        W = numpy.array([[w1], [w2], [w3], ...])
        '''
        rows, cols = A.shape
        if W_old is None:
            W_old = numpy.ones(shape=(cols, 1)) / cols
        
        gene_c = self.__gene_c()
        c = next(gene_c)
        beta = numpy.ones(shape=(rows, 1))        # 对偶变量初始化
        
        item0 = numpy.identity(cols)
        item1 = (1 + omega) * item0
        item2 = numpy.matmul(A.T, A)
        item3 = numpy.matmul(A.T, D)
        item4 = A.T
        
        W = numpy.matmul(numpy.linalg.inv(item1 + c * item2), W_old + c * item3 - numpy.matmul(item4, beta))
        beta_delta = c * (numpy.matmul(A, W) - D)
        
        while numpy.linalg.norm(beta_delta) > epsilon1 and c < epsilon2:
            beta += beta_delta
            c = next(gene_c)
            W = numpy.matmul(numpy.linalg.inv(item1 + c * item2), W_old + c * item3 - numpy.matmul(item4, beta))
            beta_delta = c * (numpy.matmul(A, W) - D)
            
        return W
            
    def __gene_c(self):
        '''
        生成阶梯状c
        '''
        c = 1
        cnt = 0
        while True:
            if cnt != 0 and cnt % 10 == 0:
                c *= 2
            yield c
            cnt += 1
            
    def solve2norm_old(self, A, D, epsilon=1.e-5, adjustable=False):
        '''
        优化问题(惯性加权方案):
        min Norm(W, 2)
        s.t. AW = D
        where:
        A = numpy.array([[r1, r2, r3, ...], [g1, g2, g3, ...], [b1, b2, b3, ...], ...])
        D = numpy.array([[r], [g], [b]])
        W = numpy.array([[w1], [w2], [w3], ...])
        '''
        rows, cols = A.shape
        
        c = 1                                  # 可调参数初始化
        beta = numpy.ones(shape=(3, 1))        # 对偶变量初始化
        
        item0 = numpy.identity(cols)
        item1 = numpy.matmul(A.T, A)
        item2 = numpy.matmul(A.T, D)
        item3 = A.T
        
        if adjustable:
            W = numpy.matmul(numpy.linalg.inv(item0 + c * item1), c * item2 - numpy.matmul(item3, beta))
            beta_delta = c * (numpy.matmul(A, W) - D)
            
            while numpy.linalg.norm(beta_delta) > epsilon and c < 1 / epsilon:
                beta += beta_delta
                c *= 1.05
                W = numpy.matmul(numpy.linalg.inv(item0 + c * item1), c * item2 - numpy.matmul(item3, beta))
                beta_delta = c * (numpy.matmul(A, W) - D)
                
            return W
        else:
            item4 = numpy.linalg.inv(item0 + c * item1)
            item5 = c * item2
            
            W = numpy.matmul(item4, item5 - numpy.matmul(item3, beta))
            beta_delta = c * (numpy.matmul(A, W) - D)
            
            while numpy.linalg.norm(beta_delta) > epsilon:
                beta += beta_delta
                W = numpy.matmul(item4, item5 - numpy.matmul(item3, beta))
                beta_delta = c * (numpy.matmul(A, W) - D)
                
            return W
        
        
        
if __name__ == '__main__':
    import sys
    sys.path.append('../parse')
    from gammaparse import DataPre
    obj = DataPre('LinkJCExcel')
    obj.load_all_files('../screens_tar', '../screens_ref')
    init, stop = obj.get_D_and_A(tar_idx=0, ref_idx_list=range(30))
    stop_D1, stop_A1 = stop[(255.0, 620.0)]
    stop_D2, stop_A2 = stop[(239.0, 620.0)]
    init_D2, init_A2 = init[(239.0, 620.0)]
    stop_D3, stop_A3 = stop[(207.0, 620.0)]
    init_D3, init_A3 = init[(207.0, 620.0)]
    
    stop_D4, stop_A4 = stop[(143.0, 620.0)]
    init_D4, init_A4 = init[(143.0, 620.0)]
    
    sol = InertiaSolver()
    W_new1 = sol.solve2norm(stop_A1, stop_D1, omega=1)
    pred1 = numpy.matmul(stop_A2, W_new1)
    delta1 = stop_D2 - pred1
    W_new2 = sol.solve2norm(stop_A2, stop_D2, omega=1)
    pred2 = numpy.matmul(stop_A3, W_new2)
    delta2 = stop_D3 - pred2
    W_new3 = sol.solve2norm(stop_A3, stop_D3, omega=1)
    pred3 = numpy.matmul(stop_A4, W_new3)
    print('new, real, curr')
    print(numpy.hstack((pred1, stop_D2, init_D2)) - stop_D2)
    print(numpy.hstack((pred2, stop_D3, init_D3)) - stop_D3)
    print(numpy.hstack((pred3, stop_D4, init_D4)) - stop_D4)
    

            
        
        
        
    
