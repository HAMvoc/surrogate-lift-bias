"""Swapped-halves study, task T4: counts of the files of the Hugging Face mirror (names and sizes only, no file is downloaded). Run from the repository root."""
import yaml, collections
from pathlib import Path
from huggingface_hub import HfApi
cfg = yaml.safe_load(open('configs/data.yaml', encoding='utf-8'))['source']
api = HfApi()
def lst(folder, ext):
    return {Path(e.path).stem: e.size for e in api.list_repo_tree(cfg['hf_repo'], path_in_repo=folder, repo_type='dataset') if e.path.endswith(ext) and getattr(e, 'size', None)}
stl = lst(cfg['stl_dir'], '.stl'); vtk = lst(cfg['vtk_dir'], '.vtk')
print('stl files', len(stl), 'vtk files', len(vtk))
def key(i):
    p = i.split('_'); return (p[0], p[1])
c = collections.Counter(); sz = collections.defaultdict(float)
for i in set(stl) & set(vtk):
    c[key(i)] += 1; sz[key(i)] += (stl[i] + vtk[i]) / 1e9
for k in sorted(c): print(k, c[k], round(sz[k], 1), 'GB', round(sz[k] / c[k] * 1000, 1), 'MB per car')
import pandas as pd
ids = pd.read_csv('data/raw/drag/DrivAerNetPlusPlus_Cd_8k_Updated.csv').ID
both = set(stl) & set(vtk)
print('Cd IDs', len(ids), 'with stl and vtk', sum(i in both for i in ids))
