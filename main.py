import time
import threading

import torch
import numpy as np

from harness import capture, control, detect
from policy import ActorCriticPolicy

def escape_and_restart():
	control.press_and_release_key(control.KEY_ESCAPE)
	time.sleep(0.5)
	control.press_and_release_key(control.KEY_RIGHT)
	control.press_and_release_key(control.KEY_RIGHT)
	control.press_and_release_key(control.KEY_ENTER)
	control.press_and_release_key(control.KEY_UP)
	control.press_and_release_key(control.KEY_ENTER)
	control.press_and_release_key(control.KEY_ENTER)

class InputController:
	def __init__(self, rate=1/8):
		self.input_rate = rate

		self.running = False

		self.current_steering = 0.0
		self.current_throttle = 0.0

		self.lock = threading.Lock()
		self.thread = None

	def start(self):
		with self.lock:
			self.current_steering = 0.0
			self.current_throttle = 0.0

			if self.running:
				return

		self.running = True
		self.thread = threading.Thread(target=self._controller_loop, daemon=True)
		self.thread.start()

	def stop(self):
		with self.lock:
			self.running = False

		if self.thread:
			self.thread.join()
			self.thread = None

	def set_actions(self, steering, throttle):
		with self.lock:
			self.current_steering = steering
			self.current_throttle = throttle

	def _controller_loop(self):
		while True:
			t_last_frame = time.perf_counter()

			with self.lock:
				if not self.running:
					break

				steering = self.current_steering
				throttle = self.current_throttle

			steering_off = 1 - abs(steering)
			throttle_off = 1 - abs(throttle)

			# The button is OFF for (1 - abs) * self.input_rate
			# Then its ON until the end of the cycle
			# Then its turned off
			# And this for two actions

			while True:
				t_passed = (time.perf_counter() - t_last_frame) / self.input_rate
				if steering_off <= t_passed:
					steering_off = t_passed * 2
					control.press_key(control.KEY_LEFT if steering < 0 else control.KEY_RIGHT)
				if throttle_off <= t_passed:
					throttle_off = t_passed * 2
					control.press_key(control.KEY_DOWN if throttle < 0 else control.KEY_UP)
				if t_passed >= 1:
					break
				time.sleep(0.001)

			control.release_key(control.KEY_LEFT if steering < 0 else control.KEY_RIGHT)
			control.release_key(control.KEY_DOWN if throttle < 0 else control.KEY_UP)

class Agent:
	SMOOTH_CAPTURE_PERF = 0.9

	MAX_TIME_WITHOUT_PROGRESS = 5

	def __init__(self, actions_per_second=3, timehack=1.0):
		self.actions_per_second = actions_per_second
		self.timehack = timehack

		self.game = detect.Harness()
		self.track = detect.TrackTracker(self.game)
		self.keys = InputController(rate=1/(8*timehack))

		self.capture_perf = None

		self.policy = ActorCriticPolicy(input_shape=(1, 96, 128))

		np.random.random()

	def lost_controls(self):
		if not self.game.is_foreground_window():
			return True
		if not self.game.get_base().get_game_has_input():
			return True
		return False

	def get_action(self, pixels):
		pixels_tensor = torch.tensor(pixels).unsqueeze(0)
		with torch.no_grad():
			mean, log_std, value = self.policy(pixels_tensor)

		std = torch.exp(log_std)
		dist = torch.distributions.Normal(mean, std)
		raw_action = dist.sample()
		action = torch.tanh(raw_action)
		log_prob_raw = dist.log_prob(raw_action).sum(dim=-1)
		log_prob = log_prob_raw - torch.log(1 - action**2 + 1e-6).sum(dim=-1)

		return action, log_prob, value

	def run_single_episode(self):
		episode = []
		discarded = False
		completed = False

		self.start_episode()
		time.sleep(-self.game.get_base().get_seconds_since_start() - 0.040)

		# Enable controls
		self.keys.start()

		prev_position = self.track.track_position()
		max_pos_achieved = prev_position
		time_of_last_progress = time.perf_counter()

		while True:
			t_frame_start = time.perf_counter()

			if self.lost_controls():
				discarded = True
				break

			pixels = capture.capture_gray()[np.newaxis, ...]
			action, logp, value = self.get_action(pixels)
			
			steering = action[0][0].item()
			throttle = action[0][1].item()
			self.keys.set_actions(steering, throttle)

			# We have to be careful, because only at this point we should measure
			# whatever will be used for our rewards, so that rewards are properly
			# accounter for by the policy.
			# but on the other hand, this creates a delay between captured pixels and
			# measured quantities (track pos, race time)
			# which /probably/ only pose problem for debugging?
			still_on_track = self.game.get_base().get_is_on_track()
			pos_on_track = self.track.track_position()
			pos_delta = pos_on_track - prev_position
			prev_position = pos_on_track
			game_t_since_start = self.game.get_base().get_seconds_since_start()
			t_now = time.perf_counter()

			print(f"s={steering:+.1f} t={throttle:+.1f} v={value[0].item():+3.1f} dP={pos_delta:+.1f}")

			# Save pixels, action and reward sources
			episode.append({
				'pixels': pixels,

				'actions': [steering, throttle],
				'logp':    logp.item(),
				'critic':  value[0].item(),

				'pos_delta': pos_delta,
				'game_time': game_t_since_start,
			})

			# Now we perform rule check, make sure we're not out of bounds
			# and we're not just chilling on same place (stuck somewhere)
			if pos_on_track > max_pos_achieved + 0.5:
				max_pos_achieved = pos_on_track
				time_of_last_progress = t_now

			if (t_now - time_of_last_progress) * self.timehack > Agent.MAX_TIME_WITHOUT_PROGRESS:
				break

			if not still_on_track:
				break

			# TODO: We're actually missing a check on whether we finished the race
			# But we're far away from there now, so I'll do that later xD

			# Now wait for next frame and start loop again
			target_frame_time = 1 / self.actions_per_second / self.timehack
			t_frame = time.perf_counter() - t_frame_start
			if t_frame < target_frame_time:
				while time.perf_counter() - t_frame_start < target_frame_time:
					time.sleep(0)
			else:
				print(f"Can't keep up! {t_frame:.3f} > {target_frame_time:.3f}")

		# Make sure to stop key presser
		self.keys.stop()

		return episode, completed, discarded

	def start_episode(self):
		# Make sure game window is active
		self.game.wait_for_window_activation()

		# If menu open - close it
		if not self.game.get_base().get_game_has_input():
			control.press_and_release_key(control.KEY_ESCAPE)

		if self.game.get_base().get_seconds_since_start() < -2.9:
			# If its race start at -3s - press enter (just to be sure)
			control.press_and_release_key(control.KEY_ENTER)
		elif self.game.get_base().get_seconds_since_start() > -0.5:
			# If the race is already ongoing - restart
			escape_and_restart()

		# Reset position tracker
		self.track.reset_position()

def apply_rewards_position(episode, last_reward=0):
	if len(episode) == 0:
		return

	for i in range(len(episode) - 1):
		episode[i]['reward'] = episode[i + 1]['pos_delta']

	episode[-1]['reward'] = last_reward

def apply_returns_reward_to_go(episode):
	for i in reversed(range(len(episode))):
		episode[i]['return'] = episode[i]['reward'] + (episode[i + 1]['return'] if i + 1 < len(episode) else 0)

agent = Agent()
optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-2)

obs = []
act = []
weights = []

while True:
	episode, completed, discarded = agent.run_single_episode()
	if discarded:
		break

	apply_rewards_position(episode)
	apply_returns_reward_to_go(episode)

	for ep in episode:
		obs.append(ep['pixels'])
		act.append(ep['actions'])
		weights.append(ep['return'])

	total_reward = sum([ep['reward'] for ep in episode])
	print(f"Added {len(episode)} trajectories with {total_reward:+5.1f} total reward. {len(obs)} trajectories now.")

	if len(obs) > 200:
		obs = torch.as_tensor(obs, dtype=torch.float32)
		act = torch.as_tensor(act, dtype=torch.float32)
		weights = torch.as_tensor(weights, dtype=torch.float32)

		optimizer.zero_grad()
		mean, log_std, value = agent.policy(obs)
		std = torch.exp(log_std)
		dist = torch.distributions.Normal(mean, std)
		raw_actions = dist.sample()
		actions = torch.tanh(raw_actions)
		entropy = dist.entropy().mean()
		log_prob_raw = dist.log_prob(raw_actions)
		log_prob_raw = log_prob_raw.sum(dim=-1)
		log_prob = log_prob_raw - torch.sum(torch.log(1 - actions**2 + 1e-6), dim=-1)
		loss = -(log_prob * weights).mean()
		loss.backward()
		optimizer.step()

		obs = []
		act = []
		weights = []
