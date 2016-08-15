#!/usr/bin/env python
from __future__ import print_function

import sys, gym
import numpy as np

#
# Test yourself as a learning agent! Pass environment name as a command-line argument.
#

env = gym.make('Sailing-v0' if len(sys.argv)<2 else sys.argv[1])

continuous_mode = False
ACTIONS = 0
try:
    ACTIONS = env.action_space.n
except AttributeError as e:
    try:
        shape = env.action_space.shape
        assert(shape==(1,) or shape==(2,))
        continuous_mode = True
    except AttributeError as e:
        print("Environment is neither Discrete or Box")
        exit()

ROLLOUT_TIME = 5000
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
human_wants_quit = False
human_x = 0.0
human_y = 0.0

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, human_wants_quit
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    if key==ord('q'):
        human_wants_quit = True
    a = key - ord('0')
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = key - ord('0')
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def mouse_motion(x, y, dx, dy):
    global human_x,human_y
    human_x = x
    human_y = y

env.reset()     # pendulum crashes without this
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
env.viewer.window.on_mouse_motion = mouse_motion

def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    for t in range(ROLLOUT_TIME):
        if not skip:
            #print("taking action {}".format(human_agent_action))
            if continuous_mode:
                # ugly but it works
                if shape==(1,):
                    a = np.array([ float(human_x) / env.viewer.window._width * \
                        (env.action_space.high[0]-env.action_space.low[0]) + env.action_space.low[0]])
                else:
                    a = np.array([float(human_x) / env.viewer.window._width * \
                                  (env.action_space.high[0] - env.action_space.low[0]) + env.action_space.low[0],
                                  float(human_y) / env.viewer.window._height * \
                                  (env.action_space.high[1] - env.action_space.low[1]) + env.action_space.low[1]]
                                 )
            else:
                a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        env.render()
        if done: break
        if human_wants_restart or human_wants_quit: break
        while human_sets_pause:
            env.render()
            import time
            time.sleep(0.1)

if continuous_mode:
    print("mouse mode: action_space={}".format(env.action_space.shape))
else:
    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

while not human_wants_quit:
    rollout(env)
