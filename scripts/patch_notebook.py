import json

path = "/home/mecha/hive_mind/notebooks/01_local_vlm_test.ipynb"
with open(path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if 'revision = "2024-03-05"' in source:
            new_source = source.replace('revision = "2024-03-05" # Using a stable revision\n', "")
            new_source = new_source.replace(', revision=revision', '')
            cell["source"] = new_source.splitlines(keepends=True)

with open(path, "w") as f:
    json.dump(nb, f, indent=1)
