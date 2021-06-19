"""Define exploration space."""
#from collections import OrderedDict
#from deephyper.benchmarks_hps.params import (activation, dropout, optimizer)
from deephyper.search.models.base import param, step, Space

epochs = param.discrete("epochs", 5, 100, step.ARITHMETIC, 1),
batch_size = param.discrete("batch_size", 8, 1024, step.GEOMETRIC, 2),

space = Space([
    #activation,
    batch_size,
    #dropout,
    epochs,
    #optimizer
])
