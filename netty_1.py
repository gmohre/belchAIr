import tensorflow as tf
import numpy as np
from util import one_hot


class Netty(object):
    def __init__(self, game_state, nlayers, nneurons, learning_rate, batch_size,
                 number_batches, buffer_size, batch_games, epsilon, test_game_state):

        self.game_state = game_state
        self.nlayers = nlayers
        self.nneurons = nneurons
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_games = batch_games
        self.number_batches = number_batches
        self.test_game_state = test_game_state

        self.sess = None

        self.action_taken_buffer = []
        self.reward_buffer = []
        self.legal_action_buffer = []
        self.state_input_buffer = []

        self.game_state_dim = len(game_state.state_space())
        self.action_dim = len(game_state.all_actions())

        self.build_model()

    def build_model(self):

        self.sess = tf.Session()
        # Placeholders
        self.state_input = tf.placeholder(tf.float32, shape=(None, self.game_state_dim), name ='state_input')
        self.legal_actions = tf.placeholder(tf.float32, shape=(None, self.action_dim), name='legal_actions')
        self.action_taken = tf.placeholder(tf.int32, shape=(None, self.action_dim), name='action_taken')
        self.reward = tf.placeholder(tf.float32, shape=(None, 1), name='reward')

        # game_state variables  and state_input hidden layer
        self.W1 = tf.Variable(tf.random_normal([self.game_state_dim, self.nneurons], stddev=0.3), name='W1')
        self.b1 = tf.Variable(tf.random_normal([self.nneurons], stddev=0.3), name='b1')
        self.hidden_layer1 = tf.nn.tanh(tf.matmul(self.state_input, self.W1) + self.b1)

        # other variables idk describe
        self.W2 = tf.Variable(tf.random_normal([self.nneurons, self.nneurons], stddev=0.3), name='W2')
        self.b2 = tf.Variable(tf.random_normal([self.nneurons], stddev=0.3), name='b2')
        self.hidden_layer2 = tf.nn.tanh(tf.matmul(self.hidden_layer1, self.W2) + self.b2)

        # action_Dim variables
        self.Wout = tf.Variable(tf.random_normal([self.nneurons, self.action_dim], stddev=0.3), name='Wout')
        self.bout = tf.Variable(tf.random_normal([self.action_dim], stddev=0.3), name='bout')
        self.out_layer = tf.matmul(self.hidden_layer2, self.Wout) + self.bout

        # zero and legal_actions mask
        self.zero = tf.constant(0, dtype=tf.float32)
        self.action_mask = tf.not_equal(self.legal_actions, self.zero)

        print('action_mask', self.action_mask.get_shape())

        # apply action_mask to out_layer
        self.out_layer_masked = tf.boolean_mask(self.out_layer, self.action_mask)
        print('out_layer_masked', self.out_layer_masked.get_shape())

        # apply action_mask to action_taken
        self.action_taken_masked = tf.boolean_mask(self.action_taken, self.action_mask)
        print('action_taken_masked', self.action_taken_masked.get_shape())

        # softmax out_layer_masked
        self.out_layer_masked_sm = tf.nn.softmax(self.out_layer_masked, name='out_masked_sm')
        print('out_layer_masked_sm', self.out_layer_masked_sm.get_shape())

        # softmax out_layer
        # why is this done here?
        self.out_layer_sm = tf.nn.softmax(self.out_layer, name='out_sm')
        print('out_layer_sm', self.out_layer_sm.get_shape())

        # mult out_layer_sm to get out_layer_sm_legal
        self.out_layer_sm_legal = self.out_layer_sm * self.legal_actions
        print('out_layer_sm_legal', self.out_layer_sm_legal.get_shape())

        # normalize out_layer_sm_legal
        self.normalization_sum = tf.reduce_sum(self.out_layer_sm_legal)
        print('normalization_sum', self.normalization_sum.get_shape())

        # mult by reciprocal of the norm_sum
        # Why not a new var?
        self.out_layer_sm_legal = tf.multiply(self.out_layer_sm_legal,
                                              tf.reciprocal(self.normalization_sum))
        print('out_layer_sm_legal', self.out_layer_sm_legal.get_shape())

        # Loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.action_taken_masked,
                                                            logits=self.out_layer_masked,
                                                            name='loss_function') * self.reward
        self.mean_loss = tf.reduce_mean(self.loss, name='mean_loss_function')

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.mean_loss)

        # Run session
        self.sess.run(tf.global_variables_initializer())

    def play_game(self):

        self.game_state.reset_game()

        reward = None
        actions_taken = []
        state_inputs = []
        legal_actions_list = []

        # reward loop
        while reward is None:
            # get legal actions
            legal_actions, all_actions = self.game_state.possible_actions()
            legal_actions_list.append(legal_actions)

            # set state_inputs
            current_game_state = self.game_state.state_space()
            state_inputs.append(current_game_state)

            # decide action to take
            policy_probs = self.sess.run(self.out_layer_sm_legal,
                                         feed_dict={self.state_input: [current_game_state],
                                                    self.legal_actions: [legal_actions]})
            action_to_take = np.argmax(policy_probs)

            # randomize policy by epsilon
            if np.random.rand() < self.epsilon:
                action_to_take = np.random.choice([index for index, action in enumerate(legal_actions) if action == 1])
            # record action taken
            # happening before action is taken
            actions_taken.append(action_to_take)

            # define reward from all_actions dict based on action_to_take key,
            # resolving in the game state
            reward = all_actions[action_to_take].resolve(self.game_state)

        # concatenate and discount rewards
        total_rewards = np.concatenate((np.zeros(len(actions_taken) - 1), [reward]))
        total_rewards_disc = discount_rewards(total_rewards)
        return actions_taken, total_rewards_disc, legal_actions_list, state_inputs

    def get_action_probabilities(self, test_game_state):
        # only really care about legal_actions here
        legal_actions, all_actions = test_game_state.possible_actions()
        current_game_state = test_game_state.state_space()

        feed_dict = {self.state_input: [current_game_state],
                     self.legal_actions: [legal_actions]}
        # assuming out_layer_sm_legal is defined?
        probabilities = self.sess.run(self.out_layer_sm_legal,
                                      feed_dict=feed_dict)
        return probabilities[0]

    def play_and_train(self):

        self.all_rewards = []

        while True:
            # Training loop
            for game_number in range(self.batch_games):
                # Play a game
                actions_taken, total_rewards_disc, legal_actions_list, state_inputs = self.play_game()
                # append rewards to all_rewards
                self.all_rewards.append(total_rewards_disc[-1])
                # append game output to the buffers
                for action_taken, total_reward_disc, legal_actions, state_input in zip(actions_taken, total_rewards_disc, legal_actions_list, state_inputs):
                    self.action_taken_buffer.append(action_taken)
                    self.reward_buffer.append(total_reward_disc)
                    self.legal_action_buffer.append(legal_actions)
                    self.state_input_buffer.append(state_input)
                    # is it time to pop the buffer?
                    if len(self.action_taken_buffer) > self.buffer_size:
                        self.action_taken_buffer.pop(0)
                        self.reward_buffer.pop(0)
                        self.legal_action_buffer.pop(0)
                        self.state_input_buffer.pop(0)
            # train
            self.train()

            # print average rewards
            print(np.mean(self.all_rewards[-self.batch_games:]))

            #print(self.test_game_state.battlefield)
            #print(self.test_game_state.tapped)

            # get action_probabilities from test_game_state
            probs = self.get_action_probabilities(self.test_game_state)

            #print(self.test_game_state.possible_actions()[0])

            # Render Probabilties
            print(probs[0], probs[1], np.sum(probs[24:26]))

            #legal_actions, all_actions = self.test_game_state.possible_actions()
            #print(all_actions[2].card_name, all_actions[2].action_name)
            #all_actions[2].resolve(self.test_game_state)
            #print(self.test_game_state.battlefield)
            #print(self.test_game_state.tapped)
            #next_probs = self.get_action_probabilities(self.test_game_state)
            #self.test_game_state.reset_game(False, True)
            #print(probs[0], np.sum(probs[2:4]), next_probs[0], np.sum(next_probs[40:44]))

    def train(self):
        # generate and randomize experience incdicies
        experience_indices = list(range(len(self.reward_buffer)))
        np.random.shuffle(experience_indices)

        # find reward mean and std
        reward_mean = np.mean(self.reward_buffer)
        reward_std = np.std(self.reward_buffer)

        # make arrays from buffers
        state_input_buffer = np.array(self.state_input_buffer)
        legal_action_buffer = np.array(self.legal_action_buffer)
        reward_buffer = np.array([[x] for x in self.reward_buffer])
        # encode action_buffer by one_hot
        action_taken_buffer = one_hot(self.action_taken_buffer, self.action_dim)

        # iterate over batch_indexes
        for batch_index in range(self.number_batches):
            if batch_index * self.batch_size < len(experience_indices):
                # calculate experience index
                experience_index = batch_index * self.batch_size: (batch_index + 1) * self.batch_size
                # get state_input from buffer
                state_input = state_input_buffer[experience_indices[experience_index]]

                # get reward from buffer
                reward = reward_buffer[experience_indices[experience_index]]

                # get legal_action from buffer
                legal_actions = legal_action_buffer[experience_indices[experience_index]]

                # get action_taken from buffer
                action_taken = action_taken_buffer[experience_indices[experience_index]]

                # normalize reward
                reward = (reward-reward_mean) / reward_std

                # run session
                self.sess.run(self.train_op,
                              feed_dict={
                                  self.state_input: state_input,
                                  self.reward: reward,
                                  self.legal_actions: legal_actions,
                                  self.action_taken: action_taken}
                              )
            #print(self.sess.run(self.mean_loss, feed_dict={self.state_input: state_input,
            #                                        self.reward: reward,
            #                                        self.legal_actions: legal_actions,
            #                                        self.action_taken: action_taken}))


"""    def train(self):
        actions_taken, total_rewards_disc, legal_actions_list, state_inputs = self.play_game()
        losses = []
        for i in range(100):
            for action_taken, disc_reward, legal_actions, state_input in zip(actions_taken, total_rewards_disc, legal_actions_list, state_inputs):
                print(disc_reward)
                #print(action_taken)
                #action_taken = tf.one_hot([action_taken], self.action_dim)
                #print(action_taken)
                self.sess.run(self.train_op, feed_dict={self.state_input: [state_input],
                                                        self.legal_actions: [legal_actions],
                                                        self.action_taken: [action_taken]})
                                                        #self.reward: np.array([disc_reward])})
                losses.append(self.sess.run(self.loss, feed_dict={self.state_input: [state_input],
                                                        self.legal_actions: [legal_actions],
                                                        self.action_taken: [action_taken]}))
                                                        #self.reward: np.array([disc_reward])}))
        return losses
"""


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * 0.9 + r[t]
        discounted_r[t] = running_add
    return discounted_r

#netty = Netty()

print(discount_rewards(np.array([0., 0., 1.])))