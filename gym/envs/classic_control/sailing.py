from __future__ import print_function

"""
https://
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time


def unit_vector(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def pt_in_rect(pt, r):
    return r[1] <= pt[0] < r[3] and r[0] <= pt[1] < r[2]


def perpendicular(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


TARGETRADIUS = 2.0
STEPS_PER_SECOND = 60
SHOAL = 10.0  # negative rewards for sailing into the shoal
SAILCOEFF = 7.0  # Newtons
RUDDER_COEFF = 0.002
MAX_ANGULAR_VELOCITY = 60.0/360.0 * 2 * np.pi / STEPS_PER_SECOND  # radians per second


class SailingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.viewer = None

        # coordinates in metres
        self.min_x = 0.0
        self.max_x = 200.0
        self.min_y = 0.0
        self.max_y = 150.0

        self.shoal_min_x = self.min_x + SHOAL
        self.shoal_max_x = self.max_x - SHOAL
        self.shoal_min_y = self.min_y + SHOAL
        self.shoal_max_y = self.max_x - SHOAL

        self.low = np.array([self.min_x, self.min_y, -1000.0, -1000.0, self.min_x, self.min_y, -1.0, -1.0])
        self.high = np.array([self.max_x, self.max_y, 1000.0, 1000.0, self.max_x, self.max_y, 1.0, 1.0])

        self.wind = np.array([0.0, -10.0]) / STEPS_PER_SECOND

        self.wind_drag = 0.0001
        self.water_drag = 0.001

        self.boat_m = 3000.0   # kg

        self.viewer = None

        self.target = np.array([(self.max_x + self.min_x) / 2, (self.max_y - self.min_y) * 0.70])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self.besttotalreward = self.totalreward = -100000.0
        self.stepnum = 0

        self.time_last_frame = time.time()
        self._seed()
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.stepnum += 1

        unit_heading = unit_vector(self.boat_heading)

        speed = np.dot(self.boat_v, unit_heading)  # positive means forward negative means backward
        if speed > 0:
            sqrtspeed = np.sqrt(np.linalg.norm(self.boat_v))
        else:
            sqrtspeed = -np.sqrt(np.linalg.norm(self.boat_v))

        # print "speed:%.3f heading:%.3f"%(speed * STEPS_PER_SECOND, self.boat_heading * 360 / (2 *np.pi))
        self.angular_velocity *= 0.95
        if action == 1:
            if self.angular_velocity<MAX_ANGULAR_VELOCITY:
                self.angular_velocity += RUDDER_COEFF * sqrtspeed
        elif action == 2:
            if self.angular_velocity>-MAX_ANGULAR_VELOCITY:
                self.angular_velocity -= RUDDER_COEFF * sqrtspeed

        # turn the boat by adjusting heading and applying a centripetal force to the centre of the turn
        self.boat_heading += self.angular_velocity
        fcentripetal = self.angular_velocity * self.boat_m

        unit_heading = unit_vector(self.boat_heading)  # new heading
        unit_perp = perpendicular(unit_heading)

        apparent_wind = self.wind - self.boat_v
        apparent_wind_speed = np.linalg.norm(apparent_wind)
        theta = angle_between(apparent_wind, -unit_heading)

        # fdrive is the force driving the boat forward which is dependent on the apparent wind angle.
        # A simple quadratic with zeros at +0.4 (23 degrees apparent) and +4.0 radians is a pretty good approximation
        # (for more info google sailing polar diagrams)
        # the absolute value o ftheta is used because it the driving force doesn't matter whether we are on
        # starboard tack (positive values of theta) or port tack (negative values of theta)
        fdrive = -(abs(theta) - 0.4) * (abs(theta) - 4.0) * apparent_wind_speed * SAILCOEFF * unit_vector(self.boat_heading)

        print("speed:%1.2f heading:%3.1f appwindangle:%3.1f appwindspeed:%1.2f fdrive:%1.2f" % \
              (speed * STEPS_PER_SECOND, self.boat_heading * 360 / (2 * np.pi), theta * 360 / (2 * np.pi),
               apparent_wind_speed * STEPS_PER_SECOND, np.linalg.norm(fdrive)))

        vforward = np.dot(self.boat_v, unit_heading) * unit_heading
        vperpendicular = self.boat_v - vforward

        # the drag force is proportional to the square of the speed in the opposite direction
        # multiplying a vector by the norm of its length efectively squares its length
        fdrag = -vforward * np.linalg.norm(vforward) * 200.0  # opposite to direction of movement
        fkeel = -vperpendicular * np.linalg.norm(vperpendicular) * 1200.0
        fperp = unit_perp * fcentripetal * np.linalg.norm(self.boat_v)

        self.boat_v += (fdrive + fdrag + fkeel + fperp) / self.boat_m
        # self.boat_v = unit_heading * ( np.linalg.norm(self.boat_v) + (np.linalg.norm(fdrive) -
        # np.linalg.norm(fdrag))/self.boat_m  )
        # print "th:%.1f drive:%.1f drag:%.1f v:%.1f"%(theta, np.linalg.norm(fdrive), np.linalg.norm(fdrag),
        # np.linalg.norm(self.boat_v))

        self.boat += self.boat_v

        previous_distance_to_target = self.distance_to_target
        self.distance_to_target = np.linalg.norm(self.boat - self.target)
        reward = previous_distance_to_target - self.distance_to_target - 0.01

        if self.boat[0] < self.shoal_min_x or self.boat[0] > self.shoal_max_x or self.boat[1] < self.shoal_min_y or \
                self.boat[1] > self.shoal_max_y:
            reward -= 0.1

        out_of_bounds = self.boat[0] < self.min_x or self.boat[0] > self.max_x or self.boat[1] < self.min_y or \
            self.boat[1] > self.max_y
        hit_target = pt_in_rect(self.boat, [self.target[1] - TARGETRADIUS, self.target[0] - TARGETRADIUS,
                                            self.target[1] + TARGETRADIUS, self.target[0] + TARGETRADIUS])

        if hit_target:
            reward += 100.0

        if out_of_bounds:
            reward -= 400.0

        done = out_of_bounds or hit_target

        self.totalreward += reward
        self.state = action

        if self.viewer is not None:
            if self.stepnum % STEPS_PER_SECOND == 0:
                self.track.append((self.boat[0], self.boat[1]))
        return np.concatenate((self.boat, self.boat_v, self.target, unit_heading)), reward, done, {}

    def _reset(self):
        #        print "Total reward:", self.totalreward
        if self.stepnum > 0 and self.totalreward > self.besttotalreward:
            self.besttotalreward = self.totalreward
            print("New Highscore: %.1f" % self.besttotalreward)
        self.totalreward = 0.0
        self.stepnum = 0

        self.boat = np.array([(self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 3])

        self.boat_v = np.array([0.0, 0.0])
        self.boat_heading = 0.5 * np.pi
        self.angular_velocity = 0.0

        self.distance_to_target = np.linalg.norm(self.boat - self.target)
        self.track = []

        return np.concatenate((self.boat, self.boat_v, self.target, unit_vector(self.boat_heading)))

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        now = time.time()
        print("fps:%.1f" % (1.0 / (now - self.time_last_frame)) )
        self.time_last_frame = now


        screen_width = 600
        scale = screen_width / (self.max_x - self.min_x)
        screen_height = int(scale * (self.max_y - self.min_y))

        boatwidth = 2.5 * scale
        boatlength = 6.0 * scale

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            target = rendering.make_circle(TARGETRADIUS * scale)
            target.set_color(0, .8, 0)
            target.add_attr(rendering.Transform(translation=(scale * self.target[0], scale * self.target[1])))
            self.viewer.add_geom(target)

            l, r, t, m, b = -boatwidth / 2, boatwidth / 2, -boatlength / 2, 0, boatlength / 2

            boat = rendering.FilledPolygon([(b, 0), (m, l), (t, l), (t, r), (m, r)])
            boat.add_attr(rendering.Transform(translation=(0, 0)))
            self.boattrans = rendering.Transform()
            boat.add_attr(self.boattrans)
            self.viewer.add_geom(boat)

        self.boattrans.set_translation(self.boat[0] * scale, self.boat[1] * scale)
        self.boattrans.set_rotation(self.boat_heading)

        track = self.viewer.draw_polyline(self.track)
        track.set_color(0.8,0.8,0.8)
        track.add_attr(rendering.Transform(scale=(scale,scale)))

        self.viewer.draw_label(self.spec.id, 7, screen_height - 25, color=(0, 0, 0, 255), font_size=20,
                               anchor_y='baseline')
        self.viewer.draw_label('episode:%d step:%d score:%.1f hi-score:%.1f' % (
            self._monitor.episode_id, self.stepnum, self.totalreward, self.besttotalreward), screen_width - 10,
            screen_height - 25, color=(0, 0, 0, 255), font_size=12, anchor_x='right',
            anchor_y='baseline')

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
