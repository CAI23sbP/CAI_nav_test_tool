import torch.nn as nn
from .Architecture_Tree import ArchitectureTree
"""
You can'not use this network, it just example how to make net.
Our simulation only uses actor-critic based model 

"""
HIDDEN_SHAPE_1 = 256
HIDDEN_SHAPE_2 = 256
HIDDEN_SHAPE_3 = 128
HIDDEN_SHAPE_4=  64
# fully connected dqn

@ArchitectureTree.register("EXAMPLE_NET")
class EXAMPLE_NET(nn.Module):
	def __init__(self, input_shape, n_actions):
		super(EXAMPLE_NET, self).__init__()

		self.sequential = nn.Sequential(nn.Linear(input_shape, HIDDEN_SHAPE_1),
										nn.ReLU(),
										nn.Linear(HIDDEN_SHAPE_1, HIDDEN_SHAPE_2),
										nn.ReLU(),
										nn.Linear(HIDDEN_SHAPE_2, HIDDEN_SHAPE_3),
										nn.ReLU(),
										nn.Linear(HIDDEN_SHAPE_3, HIDDEN_SHAPE_4),
										nn.ReLU(),
										nn.Linear(HIDDEN_SHAPE_4, HIDDEN_SHAPE_4),
										nn.ReLU(),										
										nn.Dropout(),
										nn.Linear(HIDDEN_SHAPE_4, n_actions))

	def forward(self, x):
		return self.sequential(x)