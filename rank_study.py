"""
Implements the interface to the study ranking algorithm.
"""
import time
import copy
import math
import random
import json
import os

from   typing   import List, Optional, Tuple
from   datetime import datetime, timezone

import numpy as np
import keras.layers as layers

from   keras import Model
from   keras.optimizers import adam_v2

from   aiw.pipeline.rank.types import Exam, RankModelResources
from   aiw.pipeline.rank.types import Radiologist
from   aiw.pipeline.rank.types import RankPipelineConfiguration
import aiw.logger as log


class Environment:
    def __init__(self, config: RankPipelineConfiguration):
        self.max_rads = config.max_candidates
        self.num_rads = config.max_candidates
        self.recent_exam_count = config.recent_exam_count
        self.active_user_timeout = config.active_user_timeout
        self.feature_len = config.feature_length
        self.balance_feature_length = config.balance_feature_length
        self.rad_feature_index = config.rad_feature_index
        self.rad_max_capacity = config.rad_max_capacity
        self.rvu_scale_factor = config.rvu_scale_factor
        self.exam_read_count_max = config.exam_read_count_max
        self.rad_max_rvu = config.rad_max_rvu
        self.rad_max_exams = config.rad_max_exams
        self.specialties = [str(n) for n in range(config.max_specialties)]
        self.prev_state = [np.zeros((2 + config.max_specialties, 1), dtype=np.float32), np.zeros((config.max_candidates, config.feature_length), dtype=np.float32), np.zeros((config.max_specialties, config.balance_feature_length), dtype=np.float32)]
        self.state = [np.zeros((2 + config.max_specialties, 1), dtype=np.float32), np.zeros((config.max_candidates, config.feature_length), dtype=np.float32), np.zeros((config.max_specialties, config.balance_feature_length), dtype=np.float32)]
        self.first_step = True
        self.prev_action = 0
        self.action = 0
        self.reward = 0
        self.penalty_count = 0
        self.correct_count = 0
        self.id2row = dict()
        self.row2id = dict()
        self.read_stats = config.model_resources.state_data.model_read_stats # Reference the existing dict to ensure state gets saved properly dict()
        self.last_status = dict()
        self.randomize_rows = config.randomize_rows

    @staticmethod
    def get_current_time():
        utc_now = datetime.now(timezone.utc)
        utc_date = utc_now.date()
        start_at = datetime(utc_date.year, utc_date.month, 1, tzinfo=timezone.utc)
        current_time = int((utc_now - start_at).total_seconds())
        return current_time

    # only used in simulation
    def reset(self, hard_reset=False):
        for i in range(3):
            self.prev_state[i].fill(0)
            self.state[i].fill(0)
        self.first_step = True
        self.prev_action = 0
        self.action = 0
        self.reward = 0
        self.penalty_count = 0
        self.correct_count = 0
        keys = ['specialties', 'bp_modality', 'procedures']
        for index in self.read_stats:
            if hard_reset:
                for key in keys:
                    for value in self.read_stats[index][key]:
                        self.read_stats[index][key][value]['times'] = []
                        self.read_stats[index][key][value]['mean'] = 0
                        self.read_stats[index][key][value]['std'] = 0
                        self.read_stats[index][key][value]['num_read'] = 0
                        self.read_stats[index][key][value]['num_rejected'] = 0
                self.read_stats[index]['feature'] = []
                self.read_stats[index]['hourly_rvu'] = []

            self.read_stats[index]['last_request_time'] = self.active_user_timeout
            self.read_stats[index]['active'] = False
        self.last_status.clear()

    def assign_rows(self, candidates):
        self.id2row.clear()
        self.row2id.clear()
        if self.randomize_rows:
            rows = random.sample(range(self.max_rads), len(candidates))
        else:
            rows = range(len(candidates))
        for i in range(len(candidates)):
            self.id2row[candidates[i].user_id] = rows[i]
            self.row2id[rows[i]] = candidates[i].user_id

    def get_valid_candidate(self):
        rem_cap = self.rad_max_capacity * self.prev_state[1][self.prev_action][self.rad_feature_index + 3]
        prev_difficulty = self.rvu_scale_factor * self.prev_state[0][0]
        return self.prev_state[1][self.prev_action][0] == 1# and rem_cap >= prev_difficulty

    def get_best_candidate(self):
        prio_exam = self.prev_state[0][1] == 1
        rvu = self.prev_state[1][self.prev_action][self.rad_feature_index]
        num = self.prev_state[1][self.prev_action][self.rad_feature_index + 1]
        prio = self.prev_state[1][self.prev_action][self.rad_feature_index + 2]
        if prio_exam:
            for i in range(self.max_rads):
                if self.prev_state[1][i][0] == 1:
                    prio_c = self.prev_state[1][i][self.rad_feature_index + 2]
                    if prio_c < prio:
                        return False
        else:
            for i in range(self.max_rads):
                if self.prev_state[1][i][0] == 1:
                    rvu_c = self.prev_state[1][i][self.rad_feature_index]
                    num_c = self.prev_state[1][i][self.rad_feature_index + 1]
                    if (rvu_c < rvu and num_c <= num) or (rvu_c <= rvu and num_c < num):
                        return False
        return True
    
    def get_reward(self):
        self.reward = 0

        if not self.get_valid_candidate():
            self.reward = -1
            self.penalty_count += 1
        elif self.get_best_candidate():
            self.reward = 1
            self.correct_count += 1

        mse = list()
        for i in range(len(self.specialties)):
            s = 0
            for j in range(self.balance_feature_length):
                s += self.state[2][i][j] * self.state[2][i][j]
            s /= self.balance_feature_length
            mse.append(s)
        mse = sum(mse) / len(mse)
        self.reward -= mse

        return self.reward

    def exam_feature(self, rad, exam):
        f = list()

        items = [('specialties', exam.specialty), ('bp_modality', exam.bp_modality), ('procedures', exam.procCode)]

        for (key, value) in items:
            if value not in self.read_stats[rad.user_id][key]:
                f.extend([0, 0, 0, 0])
            else:
                f.append(
                    min(1.0, self.read_stats[rad.user_id][key][value]['num_read'] / float(self.exam_read_count_max)))
                f.append(self.read_stats[rad.user_id][key][value]['mean'])
                f.append(self.read_stats[rad.user_id][key][value]['std'])
                f.append(self.get_reject_chance(rad, key, value))

        return f

    def rad_feature(self, rad):
        f = list()
        f.extend(self.get_rad_workload(rad))
        f.append(self.get_preference_rate(rad))
        f.append(self.get_reject_rate(rad.user_id))
        f.append(0)#self.get_hourly_rvu(rad.user_id))
        return f

    def add_user(self, radiologist):
        index = radiologist.user_id
        self.read_stats[index] = dict()
        self.read_stats[index]['num_updates'] = 0
        self.read_stats[index]['specialties'] = dict()
        for s in self.specialties:
            self.read_stats[index]['specialties'][s] = dict()
            self.read_stats[index]['specialties'][s]['times'] = []
            self.read_stats[index]['specialties'][s]['mean'] = 0
            self.read_stats[index]['specialties'][s]['std'] = 0
            self.read_stats[index]['specialties'][s]['num_read'] = 0
            self.read_stats[index]['specialties'][s]['num_rejected'] = 0

        self.read_stats[index]['bp_modality'] = dict()
        self.read_stats[index]['procedures'] = dict()

        self.read_stats[index]['feature'] = []
        self.read_stats[index]['hourly_rvu'] = []

        self.read_stats[index]['active'] = False
        self.read_stats[index]['last_request_time'] = self.active_user_timeout

    def update_active_users(self, radiologists):
        curr_time = self.get_current_time()

        for rad in radiologists:
            self.read_stats[rad.user_id]['last_request_time'] = curr_time

        for index in self.read_stats:
            last_request_time = self.read_stats[index]['last_request_time']
            self.read_stats[index]['active'] = True if (curr_time - last_request_time) < self.active_user_timeout else False  # self.active_user_timeout else False
            if self.read_stats[index]['active'] == False and (curr_time - last_request_time) > self.active_user_timeout*8:
                for s in self.specialties:
                    self.read_stats[index]['specialties'][s]['num_rejected'] = 0

    def update_hourly_exams(self):
        req_time = self.get_current_time()

        for rad in self.read_stats:
            keys = self.read_stats.keys()
            new_items = list()
            for item in self.read_stats[rad]['hourly_rvu']:
                if req_time - item[0] < 3600:
                    new_items.append(item)
            self.read_stats[rad]['hourly_rvu'].clear()
            self.read_stats[rad]['hourly_rvu'].extend(new_items)

    def get_hourly_rvu(self, rad):
        rvu = 0
        for item in self.read_stats[rad]['hourly_rvu']:
            rvu += item[1]
        rvu /= 60.0
        return rvu

    def update_env(self, exam, radiologists):
        self.prev_state[0] = copy.deepcopy(self.state[0])
        self.prev_state[1] = copy.deepcopy(self.state[1])
        self.prev_state[2] = copy.deepcopy(self.state[2])
        self.state[0][0] = exam.difficulty / self.rvu_scale_factor
        self.state[0][1] = exam.priority

        for index, spec in enumerate(self.specialties):
            if exam.specialty == spec:
                self.state[0][index + 2] = 1
            else:
                self.state[0][index + 2] = 0
        
        self.state[1].fill(0)
        self.assign_rows(radiologists)
        self.update_hourly_exams()

        for rad in radiologists:
            if rad.user_id not in self.read_stats:
                self.add_user(rad)
            self.last_status[rad.user_id] = rad

            self.state[1][self.id2row[rad.user_id]][0] = 1
            index = 1
            f = self.exam_feature(rad, exam)
            for i in range(len(f)):
                self.state[1][self.id2row[rad.user_id]][index] = f[i]
                index += 1
            f = self.rad_feature(rad)
            self.read_stats[rad.user_id]['feature'] = f
            for i in range(len(f)):
                self.state[1][self.id2row[rad.user_id]][index] = f[i]
                index += 1

        self.update_active_users(radiologists)
        for i in range(len(self.specialties)):
            for j in range(self.balance_feature_length):
                v = list()
                for index in self.read_stats:
                    if self.read_stats[index]['active'] and self.has_specialty(index, i):
                        v.append(self.read_stats[index]['feature'][j])
                v = np.array(v)
                self.state[2][i][j] = 0 if len(v) == 0 else np.std(v)

        return self.state

    def generate_candidates(self, exam):
        candidates = []

        for index in self.last_status:
            rad = self.last_status[index]
            if exam.specialty in rad.specialties and self.read_stats[index]['active']:
                candidates.append(rad)

        return candidates

    def get_rad_workload(self, rad):
        x = list()

        total = 0
        for e in rad.assigned_exams:
            total += e.difficulty

        x.append(min(1.0, total / self.rad_max_rvu))
        x.append(len(rad.assigned_exams) / self.rad_max_exams)
        p = 0
        for e in rad.assigned_exams:
            if e.priority == 1:
                p += 1
        x.append(p)
        x.append(rad.remaining_capacity / self.rad_max_capacity)
        return x

    def get_reject_chance(self, rad, key, value):
        if value not in self.read_stats[rad.user_id][key]:
            return 0
        elif self.read_stats[rad.user_id][key][value]['num_read'] + self.read_stats[rad.user_id][key][value]['num_rejected'] == 0:
            return 0
        else:
            return self.read_stats[rad.user_id][key][value]['num_rejected'] / (
                    self.read_stats[rad.user_id][key][value]['num_read'] + self.read_stats[rad.user_id][key][value]['num_rejected'])

    def get_preference_rate(self, rad):
        if len(rad.assigned_exams) == 0:
            return 0
        pref_rate = 0
        for exam in rad.assigned_exams:
            pref_rate += (1 - self.get_reject_chance(rad, 'specialties', exam.specialty))
        pref_rate /= len(rad.assigned_exams)
        return pref_rate

    def get_reject_rate(self, rad):
        total_read = 0
        for s in self.specialties:
            total_read += self.read_stats[rad]['specialties'][s]['num_read']
        total_rejected = 0
        for s in self.specialties:
            total_rejected += self.read_stats[rad]['specialties'][s]['num_rejected']
        total_assigned = total_read + total_rejected
        return 0 if total_assigned == 0 else total_rejected / total_assigned

    def has_specialty(self, rad, specialty):
        return specialty in self.read_stats[rad]['specialties'] and self.read_stats[rad]['specialties'][specialty]['num_read'] > 0

    def update_single(self, exam, radiologist, reject=False, completion=False, assign=False, reassign=False, read_time=None, update_rad=True):
        RAD_MIN_READ_TIMES = 100
        RAD_MAX_READ_TIMES = 200
        rad = radiologist.user_id
        items = [('specialties', exam.specialty), ('bp_modality', exam.bp_modality), ('procedures', exam.procCode)]


        if rad not in self.read_stats:
            self.add_user(radiologist)
        self.last_status[rad] = radiologist

        for (key, value) in items:
            if value is not None and value not in self.read_stats[rad][key]:
                self.read_stats[rad][key][value] = dict()
                self.read_stats[rad][key][value]['times'] = []
                self.read_stats[rad][key][value]['mean'] = 0
                self.read_stats[rad][key][value]['std'] = 0
                self.read_stats[rad][key][value]['num_read'] = 0
                self.read_stats[rad][key][value]['num_rejected'] = 0

        self.read_stats[rad]['num_updates'] = min(100, self.read_stats[rad]['num_updates'] + 1)

        if reject:
            for (key, value) in items:
                if value is not None:
                    self.read_stats[rad][key][value]['num_rejected'] += 1
        else:
            if read_time is not None and read_time > 0:
                read_time = float(read_time) / 3600.0
                for (key, value) in items:
                    if value is not None:
                        self.read_stats[rad][key][value]['times'].append(read_time)
                        while len(self.read_stats[rad][key][value]['times']) > RAD_MAX_READ_TIMES:
                            self.read_stats[rad][key][value]['times'].pop(0)
                        if len(self.read_stats[rad][key][value]['times']) < RAD_MIN_READ_TIMES:
                            normal_times = self.read_stats[rad][key][value]['times']
                        else:
                            temp_list = copy.deepcopy(self.read_stats[rad][key][value]['times'])
                            temp_list.sort()
                            q1 = temp_list[int(math.floor(0.25 * len(temp_list)))]
                            q3 = temp_list[int(math.ceil(0.75 * len(temp_list)))]
                            normal_times = [x for x in temp_list if q1 <= x <= q3]
                        mean = sum(normal_times) / len(normal_times)
                        variance = sum([((x - mean) ** 2) for x in normal_times]) / len(normal_times)
                        std = variance ** 0.5
                        self.read_stats[rad][key][value]['mean'] = mean
                        self.read_stats[rad][key][value]['std'] = std
                        self.read_stats[rad][key][value]['num_read'] += 1

                self.read_stats[rad]['hourly_rvu'].append(
                    (self.get_current_time(), exam.difficulty))
        self.update_active_users([radiologist]) 
        if update_rad:
            f = self.rad_feature(radiologist)
            self.read_stats[rad]['feature'] = f

    def get_ranking(self, action, scores, radiologists):
        self.prev_action = self.action
        self.action = action

        min_s = min(scores)
        if min_s < 0:
            scores = [max(0, x - min_s) for x in scores]

        # set scores for invalid actions to zero
        for i in range(len(scores)):
            if i not in self.row2id:
                scores[i] = 0

        # normalize scores to probability vector
        sum_s = sum(scores)
        if sum_s != 0:
            scores = [x / sum_s for x in scores]

        ranking = []
        ids = []
        for rad in radiologists:
            ranking.append(scores[self.id2row[rad.user_id]])
            ids.append(rad.user_id)
        ranking = np.argsort(ranking)[::-1]
        new_scores = []
        for index in ranking:
            conf: float = (self.read_stats[ids[index]]['num_updates'] - 50) / 50.0
            new_scores.append((ids[index], scores[self.id2row[ids[index]]], conf))

        return new_scores



class SARSAAgent:
    def __init__(self, config: RankPipelineConfiguration, environment: Environment):
        self.feature_length = config.feature_length
        self.rad_feature_index = config.rad_feature_index
        self.max_candidates = config.max_candidates
        self.max_specialties = config.max_specialties
        self.action_size = config.max_candidates
        self.state_size = [(2 + (config.max_specialties),), (config.max_candidates, config.feature_length, 1), (config.balance_feature_length,)]
        self.state_size = [(2 + config.max_specialties,), (config.max_candidates, config.feature_length, 1), (config.max_specialties, config.balance_feature_length, 1)]
        self.num_loss = 0
        self.total_loss = 0
        self.q_learning = False
        self.discount_factor = 0.5
        self.learning_rate = 0.001
        self.config_model_init_rand_prob = config.model_init_rand_prob
        self.epsilon = self.initialize_random_action_probability(config.model_resources.state_data.model_read_stats, config.model_init_rand_prob, config.site_min_exam)
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.estimate_reward = True
        self.model = config.model_resources.model_data
        self.environment = environment

    @staticmethod
    def initialize_random_action_probability(read_stats, model_init_prob, min_exams):
        _RAND_INIT_PROB = 1

        if model_init_prob == -1:  # forced initialization for bad training data (test environment or clinical study) : forced rule based distribution
            return _RAND_INIT_PROB

        if not read_stats:
            return model_init_prob

        total = 0
        keys = ['specialties', 'bp_modality', 'procedures']
        try:
            for index in read_stats:
                for key in keys:
                    for value in read_stats[index][key]:
                        total += read_stats[index][key][value]['num_read']
            return max(1 - (total / min_exams), model_init_prob) # TODO: As a potential future optimization, consider configuring the random probability based on the expected study reading size.
        except Exception as _:
            log.error("No read stats for initialization of random action probability.")
            return 1

    def build_model(self): # pragma: no cover (here for reference only; should match RankModelResources._build_model)
        input_exam = layers.Input(shape=self.state_size[0])
        input_rads = layers.Input(shape=self.state_size[1])
        input_stats = layers.Input(shape=self.state_size[2])
        inputs = [input_exam, input_rads, input_stats]

        x_rads = []
        dense = layers.Dense(self.feature_length, activation='relu')
        for i in range(self.max_candidates):
            out = layers.Lambda(lambda x: x[:, i, :, :])(input_rads)
            out = layers.Flatten()(out)
            out = dense(out)
            x_rads.append(out)

        x_stats = []
        for i in range(self.max_specialties):
            out = layers.Lambda(lambda x: x[:, i, :, :])(input_stats)
            out = layers.Flatten()(out)
            x_stats.append(out)

        x = [input_exam]
        x.extend(x_rads)
        x.extend(x_stats)
        x = layers.Concatenate()(x)

        outputs = layers.Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=adam_v2.Adam(lr=self.learning_rate))
        model.summary()
        return model
    
    def get_valid_candidate(self, state, action):
        return state[1][action][0] == 1

    def get_best_candidate(self, state, action):
        prio_exam = state[0][1] == 1
        rvu = state[1][action][self.rad_feature_index]
        num = state[1][action][self.rad_feature_index + 1]
        prio = state[1][action][self.rad_feature_index + 2]
        if prio_exam:
            for i in range(self.max_candidates):
                if state[1][i][0] == 1:
                    prio_c = state[1][i][self.rad_feature_index + 2]
                    if prio_c < prio:
                        return False
        else:
            for i in range(self.max_candidates):
                if state[1][i][0] == 1:
                    rvu_c = state[1][i][self.rad_feature_index]
                    num_c = state[1][i][self.rad_feature_index + 1]
                    if (rvu_c < rvu and num_c <= num) or (rvu_c <= rvu and num_c < num):
                        return False
        return True

    def get_action(self, state):
        random_action = np.random.rand() <= self.epsilon or self.config_model_init_rand_prob == -1
        if random_action:
            q_values = np.random.random_sample(self.action_size)
        else:
            tmp_exam = np.reshape(state[0], self.state_size[0])
            tmp_rads = np.reshape(state[1], self.state_size[1])
            tmp_stats = np.reshape(state[2], self.state_size[2])
            x_exam = np.zeros((1,) + self.state_size[0])
            x_exam[0] = tmp_exam
            x_rads = np.zeros((1,) + self.state_size[1])
            x_rads[0] = tmp_rads
            x_stats = np.zeros((1,) + self.state_size[2])
            x_stats[0] = tmp_stats
            x = [x_exam, x_rads, x_stats]
            q_values = self.model.predict(x)[0]
        return np.argmax(q_values), q_values, random_action

    def train_model(self, prev_state, prev_action, reward, state=None, action=None):
        estimate_reward = self.estimate_reward and np.random.rand() <= self.epsilon

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        tmp_prev_exam = np.reshape(prev_state[0], self.state_size[0])
        tmp_prev_rads = np.reshape(prev_state[1], self.state_size[1])
        tmp_prev_stats = np.reshape(prev_state[2], self.state_size[2])
        x_prev_exam = np.zeros((1,) + self.state_size[0])
        x_prev_exam[0] = tmp_prev_exam
        x_prev_rads = np.zeros((1,) + self.state_size[1])
        x_prev_rads[0] = tmp_prev_rads
        x_prev_stats = np.zeros((1,) + self.state_size[2])
        x_prev_stats[0] = tmp_prev_stats
        x_prev = [x_prev_exam, x_prev_rads, x_prev_stats]
        if estimate_reward:  # base discounted future reward on estimated value
            target = self.precalculate_reward(prev_state)
            target[prev_action] = reward + self.discount_factor * self.precalculate_reward(state)[action]
        else:
            target = self.model.predict(x_prev)[0]
            tmp_exam = np.reshape(state[0], self.state_size[0])
            tmp_rads = np.reshape(state[1], self.state_size[1])
            tmp_stats = np.reshape(state[2], self.state_size[2])
            x_exam = np.zeros((1,) + self.state_size[0])
            x_exam[0] = tmp_exam
            x_rads = np.zeros((1,) + self.state_size[1])
            x_rads[0] = tmp_rads
            x_stats = np.zeros((1,) + self.state_size[2])
            x_stats[0] = tmp_stats
            x = [x_exam, x_rads, x_stats]
            if self.q_learning:  # base discounted future reward on maximum possible value from current prediction
                target[prev_action] = reward + self.discount_factor * max(self.model.predict(x)[0])
            else:  # base discounted future reward on the actual action the model would take (can be random)
                target[prev_action] = reward + self.discount_factor * self.model.predict(x)[0][action]

        target = np.reshape(target, (1, self.action_size))

        loss = self.model.train_on_batch(x=x_prev, y=target)

        self.num_loss += 1
        self.total_loss += loss

    def precalculate_reward(self, state):
        target = np.zeros(self.action_size)

        # precalculate expected rewards for every possible action
        for i in range(self.action_size):
            if not self.get_valid_candidate(state, i):
                target[i] = -1
            elif self.get_best_candidate(state, i):
                target[i] = 1

        return target



class RankStudyModel:
    """
    The interface to the model used to generate a ranked list of candidate radiologists to be assigned to read a study.

    Fields
    ------
        config         : The `RankPipelineConfiguration` specifying configurable attributes associated with the model.
        environment    : The agent environment.
        agent          : The agent state.
        train          : Indicates whether this model instance also performs training.
        model          : The `RankModelResources` used to initialize the model interface. The model resources must be resident in process memory.
        last_reward    : The reward value from the prior call to `Environment.get_reward()`, updated in `RankStudyModel.get_ranking()`. Needed for model performance comparison.
        recent_exam_ids: The list of the `RankPipelineConfiguration.recent_exam_count` most recently seen exam IDs.
    """
    def __init__(self, config: RankPipelineConfiguration) -> None:
        if not config.model_resources.resident_in_memory():
            raise ValueError('RankModelResources must be resident in memory')

        self.config                     : RankPipelineConfiguration = config
        self.environment                : Environment               = Environment(config)
        self.agent                      : SARSAAgent                = SARSAAgent (config, self.environment)
        self.model                      : RankModelResources        = config.model_resources
        self.train                      : bool                      = config.train_model
        self.num_rewards                : int                       = 0
        self.last_reward                : float                     = 0
        self.total_reward               : float                     = 0
        self.recently_ranked_exams      : List[str]                 = []
        self.recently_assigned_exams    : List[Tuple]               = []
        self.recently_completed_exams   : List[str]                 = []
        self.saved_exams                : List[Tuple]               = []
        self.saved_rads                 : List[Tuple]               = []
        self.save_time                  : datetime                  = self.environment.get_current_time()


    def get_ranking(self, exam: Exam, radiologists: List[Radiologist]) -> List[Tuple[int, float, float]]:
        """
        Rank a set of candidate radiologists in decreasing order of suitability for being assigned an exam.

        Parameters
        ----------
            exam        : The study under consideration.
            radiologists: The list of candidate radiologists.

        Returns
        -------
            A list of `tuple (int, float, float)` where:
            - The first element of a tuple is the radiologist user ID,
            - The second element of a tuple is the radiologist ranking in [0, 1),
            - The third element of a tuple is the ranking confidence score in [-1, +1].
        """
        
        if exam.difficulty == 0:
            exam.difficulty = 1

        for rad in radiologists:
            for (index, e) in enumerate(rad.assigned_exams):
                if e.difficulty == 0:
                    rad.assigned_exams[index].difficulty = 1;
                    
        self.saved_exams.append((exam.specialty, exam.difficulty))
        candidates = []
        for rad in radiologists:
            assigned = []
            for e in rad.assigned_exams:
                assigned.append((e.specialty, e.difficulty))
            candidates.append((rad.user_id, rad.specialties, assigned))
        self.saved_rads.append(candidates)

        if len(self.saved_exams) >= self.config.performance_data_count:
            update_repository: bool = True if self.config.train_model else False
            self.model.update_performance_evaluation_data((self.saved_exams, self.saved_rads), update_repository)
            self.saved_exams.clear()
            self.saved_rads.clear()

        while len(self.recently_ranked_exams) >= self.config.recent_exam_count:
            self.recently_ranked_exams.pop(0)
        self.recently_ranked_exams.append(exam.eid)

        # sort radiologists by rvu
        rvu = []
        for rad in radiologists:
            rvu.append(rad.get_assigned_rvu())
        index = np.argsort(rvu)
        unused_ids = []
        for i in index[self.config.max_candidates:]:
            unused_ids.append(radiologists[i].user_id)
        new_rads = []
        for i in index[:self.config.max_candidates]:
            new_rads.append(radiologists[i])
        radiologists = new_rads

        # sort by user id
        ids = []
        for rad in radiologists:
            ids.append(rad.user_id)
        index = np.argsort(ids)
        new_rads = []
        for i in index:
            new_rads.append(radiologists[i])
        radiologists = new_rads

        self.environment.update_env(exam, radiologists)
        action, scores, random_action = self.agent.get_action(self.environment.state)
        ranking = list()
        if random_action:  # sort by priority/rvu if random action was taken
            eligible_rads = radiologists
            if exam.priority > 0:  # if a priority exam, limit candidates to those with the minimum priority exams
                prio_exams = [rad.get_num_priority() for rad in radiologists]
                eligible_rads = [rad for rad in radiologists if rad.get_num_priority() == min(prio_exams)]
                unused_ids.extend([rad.user_id for rad in radiologists if rad not in eligible_rads])
            t_special_scaled_rvu = []
            total_rad_rvu = []

            for rad in eligible_rads:
                special_rvu = rad.get_assigned_rvu_exam(exam)
                total_rad_rvu.append(rad.get_assigned_rvu())
                t_special_scaled_rvu.append(special_rvu)

            max_total_rvu = max(total_rad_rvu)
            special_scaled_rvu = []

            for rad in eligible_rads:
                special_rvu = rad.get_assigned_rvu_exam(exam)
                rejection_cnt = self.environment.read_stats[rad.user_id]['specialties'][exam.specialty]['num_rejected']
                special_penalty_rvu = exam.difficulty * rejection_cnt  # rad.get_reject_count(exam.specialty)
                if special_penalty_rvu > 0:
                    print(special_penalty_rvu)
                if rad.get_assigned_rvu() == max_total_rvu:
                    special_penalty_rvu = 0
                special_scaled_rvu.append(max(0, special_rvu - special_penalty_rvu))

            unique_values = len(np.unique(np.array(special_scaled_rvu)).tolist())
            max_rvu = max(special_scaled_rvu)
            special_scaled_rvu = [max_rvu - n + 1 for n in special_scaled_rvu]
            sum_rvu = sum(special_scaled_rvu)
            special_scaled_rvu = [n / sum_rvu for n in special_scaled_rvu]

            max_total_rvu = max(total_rad_rvu)
            total_scaled_rvu = [max_total_rvu - n + 1 for n in total_rad_rvu]
            sum_rvu = sum(total_scaled_rvu)
            total_scaled_rvu = [n / sum_rvu for n in total_scaled_rvu]



            sorted_indices = np.lexsort((total_scaled_rvu, special_scaled_rvu))

            for i in sorted_indices[::-1]:
                ranking.append((eligible_rads[i].user_id, special_scaled_rvu[i] + total_scaled_rvu[i]/5, 0))

            if unique_values == 1:  # randomly shuffle the ranking if all radiologists have the same total RVU
                np.random.shuffle(ranking)
            log.info(f'ranking based on rvu with prob={self.agent.epsilon:.5f}')
        else:
            ranking = self.environment.get_ranking(action, scores, radiologists)
            log.info(f'ranking using environment with prob={self.agent.epsilon:.5f}')

        # add candidates > 20 back into ranking with score = 0 and confidence = -1
        for i in unused_ids:
            ranking.append((i, 0, -1))
        if self.environment.first_step:
            self.environment.first_step = False
            return ranking

        reward = self.environment.get_reward() # NOTE: This changes the environment, do not move or remove.
        self.total_reward += reward
        self.num_rewards += 1
        if self.train:
            # TODO: As a possible future optimization, queue training on a background thread and return immediately.
            self.agent.train_model(self.environment.prev_state, self.environment.prev_action, reward, self.environment.state, self.environment.action)
            # TODO: As a possible future optimization, the model need not be saved with every training action.
            # Quick fix: save every 10 min
            curr_time = self.environment.get_current_time()
            if curr_time - self.save_time > 600:  # 10 min = 600, 1 min = 60
                self.model.save_model()
                self.save_time = self.environment.get_current_time()

        self.last_reward = reward
        return ranking
    
    def update_disposition(self, exam: Exam, candidate: Radiologist, rejection: bool=False, completed: bool=False, assigned: bool=False, reassigned_flag: bool=False,  work_time: Optional[int]=None) -> None:
        """
        Update the state of the environment to reflect a change in disposition of a work item.

        Parameters
        ----------
            exam     : The study whose disposition is being updated.
            candidate: The candidate radiologist associated with the change in study disposition.
            rejection: `True` if the study was rejected by the `candidate` user.
            completed: 'True' if the study was completed
            assigned: 'True' if the study was assigned
            reassigned: 'True' if the study was reassigned
            work_time: The number of seconds the user spent on the study before the disposition changed to completed; otherwise, specify `None`.
        """
        if work_time is None:
            work_time = 0

        recently_assigned_exam_ids = [e[0] for e in self.recently_assigned_exams]

        detected_reassigned = work_time == 0 and exam.eid in recently_assigned_exam_ids and exam.eid not in self.recently_completed_exams
        reassigned = detected_reassigned or reassigned_flag
        # perform rejection update for original assignee if exam was reassigned
        # version 2.1 change: only reject when the message was specified as "rejection"
        '''
        if reassigned:
            # assume the previous assignee is the last user to have been assigned this exam
            orig_candidate = [e[1] for e in self.recently_assigned_exams if e[0] == exam.eid][-1]
            self.environment.update_single(exam, orig_candidate, rejection, completed, assigned, reassigned, work_time, update_rad=False)
        '''
        if rejection:
            self.environment.update_single(exam, candidate, rejection, completed, assigned, reassigned, work_time, update_rad=False)

        manual_assignment = work_time == 0 and exam.eid not in self.recently_ranked_exams
        # perform training update for manual assignment
        if manual_assignment and not rejection:
            # generate possible candidates for manual assignment
            candidates = self.environment.generate_candidates(exam)

            # perform a simulated doRank call for manual assignment
            if len(candidates) > 1:
                _ = self.get_ranking(exam, candidates)
                log.info(f'simulated ranking with candidates number = {len(candidates)}')
        # perform updates for assignment, completion
        if not rejection:
            self.environment.update_single(exam, candidate, rejection, completed, assigned, reassigned, work_time)

        if work_time == 0:
            while len(self.recently_assigned_exams) >= self.config.recent_exam_count:
                self.recently_assigned_exams.pop(0)
            self.recently_assigned_exams.append((exam.eid, candidate))
        else:
            while len(self.recently_completed_exams) >= self.config.recent_exam_count:
                self.recently_completed_exams.pop(0)
            self.recently_completed_exams.append(exam.eid)
