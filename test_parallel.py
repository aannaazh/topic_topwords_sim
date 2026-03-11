"""
独立的测试脚本，用于在 Jupyter notebook 中运行 sim_parallel
避免循环导入和多进程问题
"""

#%%
import importlib
import sys

# 强制重新加载模块以获取最新更改
if 'sim_parallel' in sys.modules:
    importlib.reload(sys.modules['sim_parallel'])

from sim_parallel import setup_logger, run_test, run_distribution_test, run_comparison_test
from datetime import datetime


# Parameters
mode = "single"  # choices: ["single", "dist", "compare"]
runs = 30
docs = 10000
workers = 8
num_topics = 100
alpha = 0.1
beta = 0.01
sents_per_doc = 50
iterations = 20
tol = 0.001
target_param = "num_topics"
param_list_str = "2,3,5,10"

# Execution logic
unified_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

sim_params = {
    "num_topics": num_topics,
    "alpha": alpha,
    "beta": beta,
    "num_docs": docs,
    "sents_per_doc": sents_per_doc,
    "iterations": iterations,
    "num_workers": workers,
    "tol": tol,
    "use_threads": True,  # 使用线程池以支持 Jupyter notebook
}

if mode == "single":
    run_test(plot=True, timestamp=unified_timestamp, **sim_params)
elif mode == "dist":
    run_distribution_test(n_runs=runs, timestamp=unified_timestamp, **sim_params)
elif mode == "compare":
    try:
        p_list = [float(x.strip()) for x in param_list_str.split(",")]
        if all(x == int(x) for x in p_list):
            p_list = [int(x) for x in p_list]
    except ValueError:
        p_list = [x.strip() for x in param_list_str.split(",")]
        
    run_comparison_test(
        target_param=target_param, 
        param_list=p_list, 
        n_runs=runs, 
        timestamp=unified_timestamp, 
        **sim_params
    )

# %%
