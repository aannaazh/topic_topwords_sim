#!/usr/bin/env python
"""
重新加载模块并运行测试
"""
import importlib
import sys

# 如果模块已经加载，先删除
if 'sim' in sys.modules:
    del sys.modules['sim']

# 重新导入
from sim import run_test

# 运行测试
run_test(num_topics=50, alpha=0.1, beta=0.01, num_docs=10000, sents_per_doc=200, iterations=10)
