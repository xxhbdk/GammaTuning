# 数值优化模块

import numpy

class InertiaSolver(object):
    
    def solve2norm(self, A, D, epsilon=1.e-5, adjustable=False):
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
    solver = Solver()
    print(help(solver.solve2norm))




