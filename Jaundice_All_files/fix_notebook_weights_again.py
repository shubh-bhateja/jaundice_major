import nbformat

path = '/Users/shubhbhateja/College/minor_project/resnet50_detailed_pipeline.ipynb'
with open(path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and "model.fit(" in cell.source:
        # We need to replace the sample_weight line with one that satisfies both outputs
        old_str = "sample_weight={'treatment_output': np.array([dict_weights[int(label)] for label in yt_tr])},"
        new_str = "sample_weight={'treatment_output': np.array([dict_weights[int(label)] for label in yt_tr]), 'bilirubin_output': np.ones(len(yb_tr))},"
        
        cell.source = cell.source.replace(old_str, new_str)

with open(path, 'w') as f:
    nbformat.write(nb, f)
