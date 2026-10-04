import nbformat

path = '/Users/shubhbhateja/College/minor_project/resnet50_detailed_pipeline.ipynb'
with open(path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and "model.fit(" in cell.source:
        # Replace class_weight with sample_weight dynamically
        old_str = "class_weight={'treatment_output': dict_weights},"
        new_str = "sample_weight={'treatment_output': np.array([dict_weights[int(label)] for label in yt_tr])},"
        
        old_comment = "# We apply class weight to the treatment output to balance the binary classification"
        new_comment = "# We use sample_weight array instead of class_weight because Keras Multi-Output models do not support the class_weight arg"
        
        cell.source = cell.source.replace(old_str, new_str).replace(old_comment, new_comment)

with open(path, 'w') as f:
    nbformat.write(nb, f)
