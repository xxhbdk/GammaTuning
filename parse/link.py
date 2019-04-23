# 基础解析器2: 解析Link Excel, 有待完善...

import numpy
import pandas

class LinkExcel(object):
    
    def __init__(self, filename):
        self.filename = filename