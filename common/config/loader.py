import yaml

def load_config(path, cls):
    with open(path) as f:
        return cls(**yaml.safe_load(f))