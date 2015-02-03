from seq_pred import Viterbi
from seq_pred import Forward
import sys

class MTrain:
    def __init__(self, filename):
        self.num_observations = 0
        self.num_states = 0
        self.pair_list = []
        self.state_list = []
        self.obs_list = []
        self.observation_dict = {}
        self.state_dict = {}
        self.trans_mat = []
        self.emiss_mat = []
        self.list_dict = {}

        self.test_pair_list = []
        self.test_state_list = []
        self.test_obs_list = []
        self.validation_errors = []
        self.cross_validation_error = None
        self.load_file(filename)
        print self.state_dict
        print self.observation_dict

    def load_file(self, filename):
        '''Loads the data file the TAs give us. Each line contains two
        tab-delimited strings: Ron's mood (hidden states) and Ron's genre
        preference that day (observations).
        '''
        file_obj = open(filename)
        for line in file_obj.readlines():
            state, observation = line.split()
            if observation not in self.observation_dict:
                self.observation_dict[observation] = self.num_observations
                self.num_observations += 1
            if state not in self.state_dict:
                self.state_dict[state] = self.num_states
                self.num_states += 1
            obs_num = self.observation_dict[observation]
            state_num = self.state_dict[state]
            pair = [obs_num, state_num]
            self.pair_list.append(pair)
            self.state_list.append(state_num)
            self.obs_list.append(obs_num)
        # Save original data for later use (cross validation)
        self.list_dict['pair_list'] = self.pair_list
        self.list_dict['state_list'] = self.state_list
        self.list_dict['obs_list'] = self.obs_list

    def run(self):
        '''Makes the transition and emission matrices, and prints them out.'''

        print 'Running'
        self.make_trans_mat()
        self.make_emiss_mat()

    def make_trans_mat(self):
        '''Makes the transition matrix. The rows represent initial hidden
        states (moods) and the columns represent subsequent hidden states (moods).
        So as an example, self.trans_mat[i][j] is the probability that Ron
        transitions from mood i to mood j.
        '''

        # Initialize matrix
        self.trans_mat = [[0] * (self.num_states) for i in range(self.num_states)]
        for i in range(0, self.num_states):
            # Don't count last element because nothing follows it
            count = (self.state_list[:len(self.state_list) - 1]).count(i)
            for j in range(0, self.num_states):
                follow_count = 0
                for k in range(0, len(self.state_list) - 1):
                    # j follows i
                    if self.state_list[k] == i and self.state_list[k + 1] == j:
                        follow_count += 1
                value = follow_count / float(count)
                self.trans_mat[i][j] = value
        self.print_mat('Transition Matrix', self.trans_mat, True)

    def make_emiss_mat(self):
        '''Makes the emission matrix. The rows represent hidden states (moods)
        and the columns represent observations (music genres). So as an
        example, self.emiss_mat[i][j] is the probability that Ron is in mood
        i given that he is listening to genre j.
        '''

        # Initialize matrix
        self.emiss_mat = [[0] * (self.num_observations) for i in range(self.num_states)]
        for i in range(0, self.num_states):
            count = self.state_list.count(i)
            for j in range(0, self.num_observations):
                both_count = 0
                for k in range(0, len(self.pair_list)):
                    # If we have the state AND the observation
                    if self.pair_list[k][0] == j and self.pair_list[k][1] == i:
                        both_count += 1
                value = both_count / float(count)
                self.emiss_mat[i][j] = value
        self.print_mat('Emission Matrix', self.emiss_mat, True)

    def cross_validate(self):
        '''Perform 5-fold cross validation'''

        print 'Cross validate'
        test_size = len(self.pair_list) / 5
        training_size = len(self.pair_list) - test_size
        for i in range (0, 5):
            self.reset_data()
            size = test_size
            # For last slice, extend it to the end
            if i == 4:
                size += len(self.pair_list) - (i * test_size + test_size)
            self.modify_data(i * test_size, size)
            self.make_trans_mat()
            self.make_emiss_mat()
            # Get the sequence of test observations (genres) to run Viterbi on
            sequence = ''.join(map(str, self.test_obs_list))
            viterbi = Viterbi('yo.txt', self.trans_mat, self.emiss_mat, sequence)
            training_sequence = viterbi.run(sequence)
            self.add_validation_error(training_sequence)
        self.cross_validation_error = sum(self.validation_errors) / len(self.validation_errors)
        print 'Cross validation error = ', self.cross_validation_error
        return self.cross_validation_error

    def add_validation_error(self, training_sequence):
        '''Calculates and adds another error value to the list of validation errors.'''

        mismatch_count = 0
        # Get the sequence of test states (moods) to compare the results against
        test_sequence = ''.join(map(str, self.test_state_list))
        for j in range(0, len(training_sequence)):
            if training_sequence[j] != test_sequence[j]:
                mismatch_count += 1
        error = mismatch_count / float(len(training_sequence))
        print error
        self.validation_errors.append(error)

    def modify_data(self, start, size):
        '''Modifies data for cross validation.

        start - index where test set starts
        size - size of test set
        '''

        # Assuming training set will contain all observations and states.
        old_pair_list = self.pair_list
        self.pair_list = self.pair_list[0:start] + self.pair_list[start + size:]
        self.test_pair_list = old_pair_list[start:start + size]

        old_state_list = self.state_list
        self.state_list = self.state_list[0:start] + self.state_list[start + size:]
        self.test_state_list = old_state_list[start:start + size]

        old_obs_list = self.obs_list
        self.obs_list = self.obs_list[0:start] + self.obs_list[start + size:]
        self.test_obs_list = old_obs_list[start:start + size]

    def reset_data(self):
        ''' Resets data to original file inputs.'''

        self.pair_list = self.list_dict['pair_list']
        self.state_list = self.list_dict['state_list']
        self.obs_list = self.list_dict['obs_list']

    def print_mat(self, label, mat, newline):
        print '=====', label, '====='
        for row in mat:
            print row
        if newline:
            print '\n'

if __name__ == '__main__':
    mtrain = MTrain('ron.txt')
    mtrain.run()
    mtrain.cross_validate()
