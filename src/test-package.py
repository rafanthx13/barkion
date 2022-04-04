import ds_my_snippets as ds
import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')
iris = ds.reduce_mem_usage(iris)

# Esse código deve conseguir executar para que toda as funçôes estejam funcionanod, pelo menos em teoria
