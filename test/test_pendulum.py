__author__ = 'arolin'

from Viz import *
from models.Pendulum import *

model = Pendulum()
viz = Viz()

viz.init(model=model)
viz.animate()