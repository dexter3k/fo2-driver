import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticPolicy(nn.Module):
	def __init__(self, input_shape=(1, 96, 128), action_dim=2, log_std=[0.5, 0]):
		super(ActorCriticPolicy, self).__init__()

		self.input_shape = input_shape
		self.action_dim = action_dim

		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
			nn.ReLU(),
		)

		with torch.no_grad():
			x = torch.zeros(1, *input_shape)
			x = self.conv(x)
			conv_out_size = int(torch.prod(torch.tensor(x.shape[1:])))

		self.fc = nn.Sequential(
			nn.Linear(conv_out_size, 256),
			nn.ReLU(),
		)

		self.actor_mean = nn.Linear(256, action_dim)
		self.actor_log_std = nn.Parameter(torch.tensor(log_std), requires_grad=False)

		self.critic = nn.Linear(256, 1)

	def forward(self, x):
		features = self.conv(x)
		features = torch.flatten(features, start_dim=1)
		features = self.fc(features)

		mean = self.actor_mean(features)
		log_std = self.actor_log_std.expand_as(mean)

		value = self.critic(features)

		return mean, log_std, value

if __name__ == '__main__':
	policy = ActorCriticPolicy()
	dummy_input = torch.rand(1, 1, 96, 128)
	mean, log_std, value = policy(dummy_input)
	print("Mean action:", mean)  # [8, 2]
	print("Log std:", log_std)  # [8, 2]
	print("Value:", value)  # [8, 1]
