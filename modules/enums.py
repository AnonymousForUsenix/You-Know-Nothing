import enum

class eXplainMethod(enum.Enum):
    gradient = 0
    integrated_grad = 1
    guided_backprop = 2
