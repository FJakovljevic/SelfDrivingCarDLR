import os
import airsim
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import cv2
from datetime import datetime
import time


# Prevent TensorFlow from allocating the entire GPU at the start of the program.
# Otherwise, AirSim will sometimes refuse to launch, as it will be unable to 
import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

# ref: https://microsoft.github.io/AirSim/docs/image_apis/#getting-images-with-more-flexibility
def get_image(client):
    response = client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8) 
    img_rgba = img1d.reshape(response.height, response.width, 3)
    # return cv2.resize(img_rgba[75:135,0:255,:],(130,45))
    # return cv2.resize(img_rgba[70:140,0:255,:],(70,70))
    return cv2.resize(img_rgba, (86,86))


def build_dnn(number_of_actions, input_dims):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=8, activation='relu', input_shape=input_dims))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, kernel_size=4, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.3))
    model.add(Dense(number_of_actions))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model

def save_model(model):
    model.model.save('ddqn_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.h5')

def update_network_parameters(model, weights):
    model.model.set_weights(weights)

# def load_model(path):
#     return load_model(path)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)

    def store_transition(self, state, action, reward, new_state):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]

        return states, actions, rewards, new_states


# Epsilon decreasing exploration
class RandomExploration(object): 
    def __init__(self, epsilon, epsilon_dec_rate, epsilon_end):
        self.epsilon = epsilon
        self.epsilon_dec_rate = epsilon_dec_rate
        self.epsilon_min = epsilon_end

    def choose_random(self):
        rand = np.random.random()
        ret = False

        # choose if action will be random
        if rand < self.epsilon:
            ret = True

        # declining epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - 0.001 # *  self.epsilon_dec_rate
        else: 
            self.epsilon = self.epsilon_min

        return ret


# Ucitaju se kordinate (rucno mereni poceci puta preko pozicije kola u simulatoru)
def init_road_points():
    road_points = []
    with open('road_points.txt', 'r') as f:
        for line in f:
            point_values = line.split('\t')
            first_point = np.array([float(point_values[0]), float(point_values[1])])
            second_point = np.array([float(point_values[2]), float(point_values[3])])
            road_points.append(tuple((first_point, second_point)))

    return road_points

def get_reward(car_state, road_points):
    car_pos = car_state.kinematics_estimated.position
    car_point = np.array([car_pos.x_val, car_pos.y_val])

    d = 1000000
    for road in road_points:
        dist = np.abs(np.cross(road[1]-road[0],car_point-road[0])/np.linalg.norm(road[1]-road[0]))
        d = min(d, dist)

    if car_state.speed > 0.1:
        print('distance : ', d, '\texp distance factor : ', np.exp(-d*0.3), '\tlog car speed : ', np.log(car_state.speed+1))
        return np.log(car_state.speed+1)*np.exp(-d*0.3)*3
    else:
        return 0


# # Parametar options
# number_of_actions = 5             # left, hard_left, forward, hard_right, right (akcije auta)
# input_dims = (45, 130, 3)         # sizme of image (velicina slike koja se fiduje mrezi)
# gamma = 0.9                       # horizont (koliko opada vrednost nagrade u buducnosti)
# epsilon = 0.95                    # koliko posto su nasumicne akcije
# epsilon_dec_rate = 0.9999         # sa koliko se mnozi akcije, nasumicnost akcija opada sa vremenom
# epsilon_end = 0.05                # najmanje moguce epsilon
# batch_size = 100                  # velicina batch-a
# replace_target = 100              # na koliko se menjaju tezine u target mrezi
# memory_size = 100000              # velicina buffera
class Agent(object):
    def __init__(self, number_of_actions = 6, epsilon = 0.9999, batch_size = 64, input_dims = (45, 130, 3),   
            gamma = 0.9, epsilon_dec_rate = 0.99, epsilon_end = 0.5, memory_size = 10000, replace_target = 300):
        self.number_of_actions = number_of_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.replace_target = replace_target
        self.rand = RandomExploration(epsilon, epsilon_dec_rate, epsilon_end)
        self.memory = ReplayBuffer(memory_size, input_dims)
        # self.policy = build_dnn(number_of_actions, input_dims)
        # self.target = build_dnn(number_of_actions, input_dims)
        self.policy = load_model('ddqn_20200429_015522.h5')
        self.target = load_model('ddqn_20200429_015522.h5')

    # https://arxiv.org/pdf/1509.06461.pdf
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states = self.memory.sample_buffer(self.batch_size)

        # policy mreza nam izabere najbolju akciju (za novi state)
        max_actions = np.argmax(self.policy.predict(new_states), axis=1)     # [a1, a2, a3 ... a61, a61, a63]
        batch_index = np.arange(self.batch_size, dtype=np.uint8)             # [0,  1,  2  ...  61,  62,  63]

        # target mreze predvidja vrednosti sa akcijama policy mreze (za novi state)
        q_target_val = self.target.predict(new_states)
        q_target_val = q_target_val[batch_index, max_actions]

        # policy mreza predvidja vrednost za trenutni state
        q_policy_val = self.policy.predict(states)
        q_policy_val[batch_index, max_actions] = rewards + self.gamma * q_target_val       

        _ = self.policy.fit(states, q_policy_val, verbose=1)

        # ako je proslo odredjeni broj koraka zameni tezine u target mrezi
        if self.memory.mem_cntr % self.replace_target == 0:
            self.target.model.set_weights(self.policy.model.get_weights())
            save_model(self.policy)

        if self.memory.mem_cntr % 500 == 0:
            client.simPause(True)
            states, actions, rewards, new_states = self.memory.sample_buffer(500)
            max_actions = np.argmax(self.policy.predict(new_states), axis=1)
            batch_index = np.arange(500, dtype=np.uint8)
            q_target_val = self.target.predict(new_states)
            q_target_val = q_target_val[batch_index, max_actions]
            q_policy_val = self.policy.predict(states)
            q_policy_val[batch_index, max_actions] = rewards + self.gamma * q_target_val  
            _ = self.policy.fit(states, q_policy_val, epochs=50, verbose=1)
            client.simPause(False)

def interpret_action(action):
    car_controls.throttle = 1
    car_controls.brake = 0
    car_controls.steering = 0
    if action == 0:
        car_controls.steering = 0
    elif action == 1:
        car_controls.steering = -0.5
    elif action == 2:
        car_controls.steering = 0.25
    elif action == 3:
        car_controls.steering = -0.25
    elif action == 4:
        car_controls.steering = 0.5
    else:
        car_controls.throttle = 0
        car_controls.brake = 1

    return car_controls





road_points = init_road_points()

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

curent_state = get_image(client)
agent = Agent(input_dims=curent_state.shape)

while True:
    # getting reward for given state
    car_state = client.getCarState()
    reward = get_reward(car_state, road_points) 
  
    # action selection
    action = np.random.randint(agent.number_of_actions)
    rand_action = agent.rand.choose_random()
    if not rand_action:
        action = agent.policy.predict(curent_state.reshape(1, *curent_state.shape))
        action = np.argmax(action)

    # controling the car
    car_controls = interpret_action(action)
    client.setCarControls(car_controls)

    if rand_action:
        print('Reward: ', reward, '\t Action: ', action, '\t Random action, epsilon: ', agent.rand.epsilon)
    else:
        print('Reward: ', reward, '\t Action: ', action, '\t Choosen action')
    # getting new state
    new_state = get_image(client)

    # storing in replay_buffer
    agent.memory.store_transition(curent_state, action, reward, new_state)

    # check if in new_state there is collision, if so we reset car and give huge negativ reward
    collision_info = client.simGetCollisionInfo()
    if collision_info.has_collided:
        agent.memory.store_transition(new_state, action, -100, new_state)
        client.reset()
        # car_controls.brake = 0
        car_controls.throttle = 1
        # car_controls.steering = 0
        client.setCarControls(car_controls)
        time.sleep(0.5)

    agent.learn()
    curent_state = new_state


client.reset()
client.enableApiControl(False)