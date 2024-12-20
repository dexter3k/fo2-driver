import time
import threading

import numpy as np

from harness import capture, control, detect

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

		np.random.random()

	def lost_controls(self):
		if not self.game.is_foreground_window():
			return True
		if not self.game.get_base().get_game_has_input():
			return True
		return False

	def capture_and_measure(self):
		t_before = time.perf_counter()
		pixels = capture.capture_gray(out_size=(128, 96))
		t_total = time.perf_counter() - t_before

		if self.capture_perf is None:
			self.capture_perf = t_total
		else:
			s = Agent.SMOOTH_CAPTURE_PERF
			self.capture_perf = self.capture_perf * s + t_total * (1 - s)

		return pixels

	def run_single_episode(self):
		episode = []
		discarded = False
		premature = False

		self.start_episode()
		time.sleep(-self.game.get_base().get_seconds_since_start() - 0.040)

		# Enable controls
		self.keys.start()

		max_pos_achieved = self.track.track_position()
		time_of_last_progress = time.perf_counter()

		while True:
			t_frame_start = time.perf_counter()

			if self.lost_controls():
				discarded = True
				break

			pixels = self.capture_and_measure()

			# action, value, logp = self.get_action(pixels)
			self.keys.set_actions(np.random.random() * 2 - 1, 0.9)

			# We have to be careful, because only at this point we should measure
			# whatever will be used for our rewards, so that rewards are properly
			# accounter for by the policy.
			# but on the other hand, this creates a delay between captured pixels and
			# measured quantities (track pos, race time)
			# which /probably/ only pose problem for debugging?
			still_on_track = self.game.get_base().get_is_on_track()
			pos_on_track = self.track.track_position()
			game_t_since_start = self.game.get_base().get_seconds_since_start()
			t_now = time.perf_counter()

			# Save pixels, action and reward sources
			episode.append({
				'pixels': pixels,

				'actions': [0, 0],
				'logp':    0,
				'critic':  0,

				'position':  pos_on_track,
				'game_time': game_t_since_start,
			})

			# Now we perform rule check, make sure we're not out of bounds
			# and we're not just chilling on same place (stuck somewhere)
			if pos_on_track > max_pos_achieved + 0.5:
				max_pos_achieved = pos_on_track
				time_of_last_progress = t_now

			if (t_now - time_of_last_progress) * self.timehack > Agent.MAX_TIME_WITHOUT_PROGRESS:
				premature = True
				break

			if not still_on_track:
				premature = True
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

		print(f"Trajectory of {len(episode)} steps. premature={premature} discarded={discarded}")
		return episode, premature, discarded

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

agent = Agent()
discarded = agent.run_single_episode()
