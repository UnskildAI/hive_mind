ACTION_REGISTRY = {}

def register(name):
    def wrapper(cls):
        ACTION_REGISTRY[name] = cls
        return cls
    return wrapper


### used like
# @register("flow_policy")
# class FlowPolicy:
    # ...