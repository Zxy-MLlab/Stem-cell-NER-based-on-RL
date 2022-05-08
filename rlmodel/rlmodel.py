import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import random
import tqdm
import json
import os
import shutil
import do_devel

def get_action(prob):  #根据概率值随机选择动作

    tmp = prob[0]
    result = np.random.rand()
    if result>0 and result< tmp:
        return 1
    elif result >=tmp and result<1:
        return 0

def decide_action(prob):  #根据概率值大小选择动作
    tmp = prob[0]
    if tmp>=0.5:
        return 1
    elif tmp < 0.5:
        return 0


def get_reward(unselect_sentence_label_list, select_sentence_label_list, true_sentence_label_list, r):
    s_TP, t_TP, s_FP = 0, 0, 0
    for sentence_label_temp in select_sentence_label_list:
        sentence = sentence_label_temp['sentence']
        pre_label = sentence_label_temp['label']
        true_label = []
        for true_sentence_temp in true_sentence_label_list:
            if sentence == true_sentence_temp['sentence']:
                true_label = true_sentence_temp['label']
                break
        if pre_label == true_label:
            s_TP += 1
        else:
            s_FP += 1
    if s_TP+s_FP == 0:
        precision = 0.0
    else:
        precision = float(s_TP) / float(s_TP+s_FP)
    for unselect_sentence_label_temp in unselect_sentence_label_list:
        sentence = unselect_sentence_label_temp['sentence']
        pre_label = unselect_sentence_label_temp['label']
        true_label = []
        for true_sentence_temp in true_sentence_label_list:
            if sentence == true_sentence_temp['sentence']:
                true_label = true_sentence_temp['label']
                break
        if pre_label == true_label:
            t_TP += 1
    if t_TP == 0:
        recall = 0.0
    else:
        recall = float(s_TP) / float(t_TP)
    #r = 0.0
    f1 = r*precision + (1-r)*recall
    reward = f1
    return reward

class environment():  #定义环境类

    def __init__(self,sentence_len):
        self.sentence_len = sentence_len  #定义句子长度：Max_length = 256


    def reset(self,batch_sentence_ebd,batch_label_ebd,batch_reward):  #重置环境
        self.batch_reward = batch_reward  #当前batch的奖励
        self.batch_len = len(batch_sentence_ebd)  #当前batch中句子的个数
        self.sentence_ebd = batch_sentence_ebd  #当前batch中所有句子的向量表示
        self.label_ebd = batch_label_ebd  #当前batch中所有句子标签词向量表示
        self.current_step = 0  #当前正在选择该batch中的第几个句子
        self.num_selected = 0  #当前已经挑选该batch中的句子个数
        self.current_step = 0
        self.list_selected = []  #当前该batch中已挑选的句子
        self.vector_current = self.sentence_ebd[self.current_step]  #当前正在选择该batch中的句子的向量表示
        self.label_current = self.label_ebd[self.current_step]  #当前正在选择改batch中的句子标签词向量表示
        self.vector_mean = np.array([0.0 for x in range(self.sentence_len)], dtype=np.float32)
        self.vector_sum = np.array([0.0 for x in range(self.sentence_len)], dtype=np.float32)

        current_state = [self.vector_current,self.vector_mean, self.label_current]  #当前状态表示：1：当前正在选择该batch中的句子的向量表示，2：当前已选择该batch中所有向量的和，3：当前正在选择的句子标签的的向量表示
        return current_state


    def step(self,action):

        if action == 1:  #如果选择的行为a=1
            self.num_selected +=1  #已选择句子数量加1
            self.list_selected.append(self.current_step)  #将当前正在挑选的句子加入已选择句子列表

        self.vector_sum =self.vector_sum + action * self.vector_current #更新当前已选择句子词向量的和
        if self.num_selected == 0:  #如果当前已选句子个数为0
            self.vector_mean = np.array([0.0 for x in range(self.sentence_len)], dtype=np.float32)
        else:
            self.vector_mean = self.vector_sum / self.num_selected  #更新当前已选择句子的向量的平均值

        self.current_step +=1  #当前正在选择的句子下标加1

        if (self.current_step < self.batch_len):  #如果当前正在选择的句子下标小于整个batch中句子的长度
            self.vector_current = self.sentence_ebd[self.current_step]  #更行当前正在选择的句子的向量表示
            self.label_current = self.label_ebd[self.current_step]  #更新当前正在选择的句子标签词向量表示
        current_state = [self.vector_current, self.vector_mean, self.label_current] #更新当前状态的表示
        return current_state

    def reward(self):
        assert (len(self.list_selected) == self.num_selected)  #判断当前已选句子的数量是否等于选择的个数
        reward = [self.batch_reward[x] for x in self.list_selected]  #获取当前所有已选择的句子的奖励值
        reward = np.array(reward)  #将当前已获得的奖励列表转换成arrary
        reward = np.sum(reward)  #对当前已获得的所有奖励值取平均值
        return reward

class agent():
    def __init__(self, lr,s_size,l_size):

        self.state_in = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        self.label = tf.placeholder(shape=[None,l_size], dtype=tf.float32)

        self.input = tf.concat(axis=1,values = [self.state_in, self.label])

        self.prob = tf.reshape(layers.fully_connected(self.input,1,tf.nn.sigmoid), [-1])

        #compute loss
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        #the probability of choosing 0 or 1
        self.pi = self.action_holder * self.prob + (1 - self.action_holder) * (1 - self.prob)

        #loss
        self.loss = -tf.reduce_sum(tf.log(self.pi) * self.reward_holder)

        # minimize loss
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

        #待优化的参数
        self.tvars = tf.trainable_variables()

        #manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


        #compute gradient
        self.gradients = tf.gradients(self.loss, self.tvars)

        #update parameters using gradient
        self.gradient_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, self.tvars))

def train(gamma):
    true_sentence_label_list = []
    with open('data/train_true.json', 'r', encoding='utf-8') as f_read:
        while True:
            data_line = f_read.readline()
            if not data_line:
                break
            else:
                data_json = json.loads(data_line)
                true_sentence_label_list.append(data_json)
    all_sentence_ebd = np.load('data/all_train_sentence_ebd.npy', allow_pickle=True)
    all_label_ebd = np.load('data/all_train_label_ebd.npy', allow_pickle=True)
    all_reward = np.load('data/all_train_reward.npy', allow_pickle=True)
    all_sentence_label = np.load('data/all_train_sentence_label.npy', allow_pickle=True)

    average_reward = np.array(0.0)
    print(average_reward)

    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl)
    env = environment(256)

    with g_rl.as_default():
        with sess2.as_default():

            myAgent = agent(0.03,512,50)
            updaterate = 1
            num_epoch = 10  # 训练次数：10
            sampletimes = 5  # 采样次数：5
            best_reward = -100000

            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()

            tvars_best = sess2.run(myAgent.tvars)
            for index, var in enumerate(tvars_best):
                tvars_best[index] = var * 0
            tvars_old = sess2.run(myAgent.tvars)
            gradBuffer = sess2.run(myAgent.tvars)
            for index, grad in enumerate(gradBuffer):
                gradBuffer[index] = grad * 0
            g_rl.finalize()  # 结束当前的会话图，使之成为只读

            for epoch in range(num_epoch):
                all_list = list(range(len(all_sentence_ebd)))  # 获取all_sentence_ebd的列表下标值
                total_reward = []
                random.shuffle(all_list)  # 随机打乱所有句子的下标值
                chosen_size = 0
                all_chosen_reward = []

                for batch in tqdm.tqdm(all_list):  # 一次eposide训练，给每次eposide训练添加进度条
                    batch_sentence_ebd = all_sentence_ebd[batch]  #一个batch中所有句子的词向量
                    batch_label_ebd = all_label_ebd[batch]
                    batch_reward = all_reward[batch]  #一个batch中所有奖励
                    batch_sentence_label = all_sentence_label[batch]
                    batch_len = len(batch_sentence_ebd)  #一个batch长度

                    list_list_state = []  # batch层次的状态列表
                    list_list_action = []  # batch层次的行动列表
                    list_list_reward = []  # batch层次的奖励列表
                    avg_reward = 0

                    # 采样：对一个batch进行sampletimes次采样
                    for j in range(sampletimes):
                        state = env.reset(batch_sentence_ebd=batch_sentence_ebd, batch_label_ebd=batch_label_ebd, batch_reward=batch_reward)
                        list_action = []  # 针对一个batch进行一次采样的行为列表
                        list_state = []  # 针对一个batch进行一次采样的状态列表
                        select_sentence_label = []
                        old_prob = []

                        #对一个batch进行采样
                        for i in range(batch_len):
                            state_in = np.append(state[0], state[1], axis=0)
                            label = state[2]
                            feed_dict = {}
                            feed_dict[myAgent.state_in] = [state_in]
                            feed_dict[myAgent.label] = label
                            prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                            old_prob.append(prob[0])
                            action = get_action(prob)
                            if action == 1:
                                select_sentence_label.append(batch_sentence_label[i])
                            list_action.append(action)  # 将当前选择的动作加入动作列表
                            list_state.append(state)  # 将当前的状态加入状态列表
                            state = env.step(action)  # 根据当前的状态及动作选择下一个状态

                        if env.num_selected == 0:  #如果个体选择句子，则即时奖励为平均奖励，否则则为环境给的奖励
                            tmp_reward = average_reward
                        else:
                            tmp_reward = env.reward()
                        tmp_reward += get_reward(unselect_sentence_label_list=batch_sentence_label, select_sentence_label_list=select_sentence_label, true_sentence_label_list=true_sentence_label_list, r=gamma)


                        avg_reward += tmp_reward  # 个体奖励加上即时奖励
                        list_list_state.append(list_state)  # 将采样完成一个batch中的状态存储到eposide层的状态列表中
                        list_list_action.append(list_action)  # 将采样完成一个batch中的个体选择的行为存储到eposide层的状态中
                        list_list_reward.append(tmp_reward)

                    avg_reward = avg_reward / sampletimes

                    # 对之前的采样结果计算梯度值
                    for j in range(sampletimes):
                        # 拿到一次采样的数据：{state1, action1, ... , reward1}
                        list_state = list_list_state[j]  # 拿到其中一次采样的状态
                        list_action = list_list_action[j]  # 拿到其中一次采样的行动
                        reward = list_list_reward[j]  # 拿到其中一次采样的奖励

                        # compute gradient
                        # start = time.time()
                        list_reward = [reward - avg_reward for x in range(batch_len)]
                        list_state_in = [np.append(state[0], state[1], axis=0) for state in list_state]  # 当前一次采样的句子及已经选择的句子状态表示
                        list_label = [state[2][0] for state in list_state]

                        feed_dict = {}
                        feed_dict[myAgent.state_in] = list_state_in
                        feed_dict[myAgent.label] = list_label
                        feed_dict[myAgent.reward_holder] = list_reward
                        feed_dict[myAgent.action_holder] = list_action
                        grads = sess2.run(myAgent.gradients, feed_dict=feed_dict)
                        for index, grad in enumerate(grads):
                            gradBuffer[index] += grad

                    state = env.reset(batch_sentence_ebd=batch_sentence_ebd, batch_label_ebd=batch_label_ebd, batch_reward=batch_reward)
                    old_prob = []
                    select_sentence_label = []
                    for i in range(batch_len):
                        state_in = np.append(state[0], state[1], axis=0)
                        label = state[2]
                        feed_dict = {}
                        feed_dict[myAgent.state_in] = [state_in]
                        feed_dict[myAgent.label] = label
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                        old_prob.append(prob[0])
                        action = decide_action(prob)
                        if action == 1:
                            select_sentence_label.append(batch_sentence_label[i])
                        state = env.step(action)

                    chosen_reward = [batch_reward[x] for x in env.list_selected]
                    all_chosen_reward += chosen_reward
                    chosen_reward.append(get_reward(unselect_sentence_label_list=batch_sentence_label, select_sentence_label_list=select_sentence_label, true_sentence_label_list=true_sentence_label_list, r=gamma))
                    total_reward += chosen_reward
                    chosen_size += len(env.list_selected)
                    # print(chosen_reward)

                feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                sess2.run(myAgent.update_batch, feed_dict=feed_dict)
                for index, grad in enumerate(gradBuffer):
                    gradBuffer[index] = grad * 0

                # get tvars_new
                tvars_new = sess2.run(myAgent.tvars)

                # update old variables of the target network
                tvars_update = sess2.run(myAgent.tvars)
                for index, var in enumerate(tvars_update):
                    tvars_update[index] = updaterate * tvars_new[index] + (1 - updaterate) * tvars_old[index]

                feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_update))
                sess2.run(myAgent.update_tvar_holder, feed_dict)
                tvars_old = sess2.run(myAgent.tvars)

                # find the best parameters
                total_reward = np.mean(np.array(total_reward))

                if (total_reward > best_reward):
                    best_reward = total_reward
                    tvars_best = tvars_old
                TP,TF = 0,0
                for opt in all_chosen_reward:
                    if opt == 0.1:
                        TP = TP + 1
                    else:
                        TF = TF + 1
                precision = float(TP) / float(TP + TF)
                print("precision:", precision)
                print('chosen sentence size:', chosen_size)
                print('total_reward:', total_reward)
                print('best_reward', best_reward)

            # set parameters = best_tvars
            feed_dict = dictionary = dict(zip(myAgent.tvars_holders, tvars_best))
            sess2.run(myAgent.update_tvar_holder, feed_dict)
            # save model
            saver.save(sess2, save_path='rlmodel/origin_rl_model.ckpt')

    return

def select(save_path, x):
    #正确标签测试数据集
    if x == 'precise':
        train_sentence = np.load('test_data/sentence_label.npy', allow_pickle=True)

        all_sentence_ebd = np.load('test_data/all_sentence_ebd.npy', allow_pickle=True)
        all_label_ebd = np.load('test_data/all_label_ebd.npy', allow_pickle=True)
        all_reward = np.load('test_data/reward_batch.npy', allow_pickle=True)

    #训练数据集
    if x == 'train':
        train_sentence = np.load('data/all_train_sentence_label.npy', allow_pickle=True)

        all_sentence_ebd = np.load('data/all_train_sentence_ebd.npy', allow_pickle=True)
        all_label_ebd = np.load('data/all_train_label_ebd.npy', allow_pickle=True)
        all_reward = np.load('data/all_train_reward.npy', allow_pickle=True)

    #测试数据集
    if x == 'test':
        train_sentence = np.load('data/all_test_sentence_label.npy', allow_pickle=True)

        all_sentence_ebd = np.load('data/all_test_sentence_ebd.npy', allow_pickle=True)
        all_label_ebd = np.load('data/all_test_label_ebd.npy', allow_pickle=True)
        all_reward = np.load('data/all_test_reward.npy', allow_pickle=True)

    selected_sentence_label = []

    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl)
    env = environment(256)

    with g_rl.as_default():
        with sess2.as_default():

            myAgent = agent(0.03, 512, 50)
            init = tf.global_variables_initializer()
            sess2.run(init)
            saver = tf.train.Saver()
            saver.restore(sess2, save_path=save_path)
            g_rl.finalize()

            for epoch in range(1):

                total_reward = []
                num_chosen = 0

                all_list = list(range(len(all_sentence_ebd)))

                for batch in tqdm.tqdm(all_list):

                    batch_sentence_label = train_sentence[batch]
                    batch_sentence_ebd = all_sentence_ebd[batch]
                    batch_label_ebd = all_label_ebd[batch]
                    batch_reward = all_reward[batch]
                    batch_len = len(batch_sentence_ebd)

                    # reset environment
                    state = env.reset(batch_sentence_ebd=batch_sentence_ebd, batch_label_ebd=batch_label_ebd, batch_reward=batch_reward)
                    old_prob = []

                    for i in range(batch_len):
                        state_in = np.append(state[0], state[1])
                        label = state[2]
                        feed_dict = {}
                        feed_dict[myAgent.state_in] = [state_in]
                        feed_dict[myAgent.label] = label
                        prob = sess2.run(myAgent.prob, feed_dict=feed_dict)
                        old_prob.append(prob[0])
                        action = decide_action(prob)
                        # produce data for training cnn model
                        state = env.step(action)
                        if action == 1:
                            num_chosen+=1
                            selected_sentence_label.append(batch_sentence_label[i])
                    print("chosen num:", num_chosen)
    output_path = 'output/sentence_label' + '(' + str(x) + ')' + '.json'
    with open(output_path, 'a', encoding='utf-8') as f_write:
        for sentence_label_temp in selected_sentence_label:
            json.dump(sentence_label_temp, f_write)
            f_write.write('\n')
    f_write.close()
    return

def main():
    print("train rlmodel")
    train(gamma=0.5)

    print("writing select sentence")
    select('rlmodel/origin_rl_model.ckpt', x='precise')


if __name__ == '__main__':
    main()