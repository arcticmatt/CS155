import sys
import math
import numpy as np
import random
from seq_pred_mod import Viterbi

print_backward = False
print_forward = False

'''
Overview of my implementation (copied from our group email to you):

Instead of implementing gradient descent with the new notation using a stacked
weight vector of 'w's and a stacked feature vector of 'phi's, we instead decided
to implement gradient descent using the scoring function, where our 'weight vector'
were our emission and transition matrices. Thus, when we did gradient descent,
instead of looking to 'phi' to see which element of our weight vector to update,
we directly updated the correct emission and transition entries based on the current
states and emission we were considering (a, b, j). Here's some very short 'pseudocode':

for (j goes through entire sequence)
     calculate Z(x) denominator
     for (each state A)
          for (each state B)
               calculate logZ gradient (alpha * G * beta), and divide by Z(x) denominator
               shift (A, B) transition
               shift (A, observation at J) emission

For the second term (the F gradient), we simply subtracted 1 from the shift matrix
for every transition and every state-emission combination we found in the dataset (similar to above).

We also altered Viterbi to sum instead of multiply in order to test our algorithm in the DP process.

I also added in a dynamic normalization factor to avoid underflow/overflow, and
used uniform initial weights (for my emission and transition entries) between
0 and 1.
'''

class CRF:
    def __init__(self, filename):
        '''Constructor for the CRF class. This class uses the Forward
        and backward algorithms along with gradient descent to train
        a model for sequence prediction. The model is given by transition
        and emission matrices. This class can also perform cross validation
        with Viterbi to gauge performance.
        '''

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

        self.load_file(filename)

        self.state_seq = ''.join(map(str, self.state_list))
        self.obs_seq = ''.join(map(str, self.obs_list))

        # Cross validation vars
        self.cross_val_slice = 0
        self.test_pair_list = []
        self.test_state_list = []
        self.test_obs_list = []
        self.validation_errors = [sys.maxint for x in range(5)]
        self.cross_validation_error = None
        print 'Done initializing CRF'

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

    def run_gradient_descent(self, is_cross_val = False):
        '''Performs gradient descent to minimize the negative log loss over the
        training set, which in our case is just one sequence.
        '''

        # Initialize matrices for the observation sequence
        self.matrices = Matrices(self.num_states, self.num_observations, self.obs_seq)
        self.forward = Forward(self.num_states, self.matrices)
        self.backward = Backward(self.num_states, self.matrices)

        print '======================================================================================'
        print '======================================================================================'
        print '======================================================================================'
        print '==================================== RUNNING ========================================='
        print '======================================================================================'
        print '======================================================================================'
        print '======================================================================================'
        # Runs either 500 epochs, or less if we stabilize before then (which usually
        # happens). Or, we break if we underflow or overflow. However, underflow
        # or overflow should not occur, as we use specifically chosen C_alpha and
        # C_beta values to prevent this. It is more a precaution than anything else.
        for i in range(0, 500):
            print '========== Epoch', i, '=========='
            self.matrices.init_shift_trans_mat()
            self.matrices.init_shift_emiss_mat()

            self.forward.run(self.obs_seq)
            self.backward.run(self.obs_seq)

            # Calculates the gradient step
            self.compute_gradient()

            # We want to break if we underflow or overflow
            if self.matrices.shift_has_inf() or self.matrices.shift_has_nan():
                break

            if not self.matrices.is_changing():
                break

            # Applies the gradient step
            self.matrices.update_trans_mat()
            self.matrices.update_emiss_mat()

            # Compute and possibly add the current error
            if is_cross_val:
                error = self.add_validation_error()
                print 'Error =', error

        print 'Done with gradient descent'

    def compute_gradient(self):
        '''Computes the gradient for one training point. The methods that this
        method calls will update the shift matrices in the Matrices class so that
        by the time this method is done, we will have calculated how much to shift
        our "weight vector" (transition and emission matrices) by for the current
        epoch.'''

        # Loop through positions in sequence
        sum_gradient = 0
        # For seq_pos = 0, there is no transition, so the only thing we want
        # to do is update our shift_emiss_mat
        self.matrices.update_shift_emiss_mat(int(self.state_seq[0]), 0, -1)
        for seq_pos in range(1, len(self.obs_seq)):
            # Z-score is the same for every a,b (given a position in the sequence)
            z_score = self.compute_z_score(seq_pos)
            z_gradient = self.compute_z_gradient(seq_pos, z_score)
            f_gradient = self.compute_f_gradient(seq_pos)
            gradient = f_gradient + z_gradient
            sum_gradient += gradient
        return sum_gradient

    def compute_f_gradient(self, seq_pos):
        '''Computes f-gradient for a given position in the sequence.'''

        # Update trans mat
        state_curr = int(self.state_seq[seq_pos])
        state_prev = int(self.state_seq[seq_pos - 1])
        self.matrices.update_shift_trans_mat(state_curr, state_prev, -1)
        # Update emiss mat
        self.matrices.update_shift_emiss_mat(state_curr, seq_pos, -1)
        return -2

    def compute_z_gradient(self, seq_pos, z_score):
        '''Computes z-gradient for a given position in the sequence.'''

        z_gradient = 0
        for state_prev in range(0, self.num_states):
            for state_curr in range(0, self.num_states):
                single_z_gradient = self.compute_single_z_gradient(state_curr, state_prev, seq_pos, z_score)
                z_gradient += single_z_gradient
        return z_gradient

    def compute_single_z_gradient(self, state_curr, state_prev, seq_pos, z_score):
        '''Helper function to compute the gradient.
        Computes the z-gradient for a specific current state, previous state,
        sequence position, and z-score.
        '''

        seq_int = int(self.obs_seq[seq_pos])
        forward_score = self.forward.get_val(state_prev, seq_pos - 1)
        backward_score = self.backward.get_val(state_curr, seq_pos)
        g_score = self.matrices.get_g_score_forward(seq_pos, state_curr, state_prev, seq_int)
        numerator = forward_score * g_score * backward_score
        ans = numerator / float(z_score)
        # Update "weight vector" (e.g. transition and emission matrices)
        self.matrices.update_shift_trans_mat(state_curr, state_prev, ans)
        self.matrices.update_shift_emiss_mat(state_curr, seq_pos, ans)
        return ans

    def compute_z_score(self, seq_pos):
        '''Helper function to compute the Z-gradient.
        This method computes the denominator in the gradient of the log of Z(x).
        '''

        z_score = 0
        seq_int = int(self.obs_seq[seq_pos])
        # Sum over all states
        for state_prev in range(0, self.num_states):
            for state_curr in range(0, self.num_states):
                forward_score = self.forward.get_val(state_prev, seq_pos - 1)
                backward_score = self.backward.get_val(state_curr, seq_pos)
                g_score = self.matrices.get_g_score_forward(seq_pos, state_curr, state_prev, seq_int)
                score = forward_score * g_score * backward_score
                z_score += score
        return z_score

    def cross_validate(self):
        '''Perform 5-fold cross validation'''

        test_size = len(self.pair_list) / 5
        training_size = len(self.pair_list) - test_size
        for i in range (0, 5):
            self.cross_val_slice = i
            self.reset_data()
            size = test_size
            # For last slice, extend it to the end
            if i == 4:
                size += len(self.pair_list) - (i * test_size + test_size)
            self.modify_data(i * test_size, size)
            self.run_gradient_descent(True)
        self.cross_validation_error = sum(self.validation_errors) / len(self.validation_errors)
        print 'Cross validation error = ', self.cross_validation_error
        print 'The array of errors was', self.validation_errors
        return self.cross_validation_error

    def add_validation_error(self):
        '''Computes the out of sample error for the current training set,
        test set, and the current set of transition and emission matrices.

        Has the following pocketing functionality:
        If this error is less than the error we currently have for the cross_val_slice (which
        runs from 0-4 since we are doing 5-fold cross validation), we will store it in
        our array of validation errors. This essentially means we are pocketing the
        minimum errors for each slice while we do cross validation.
        But it is currently commented out to go for a more realistic cross validation
        error as opposed to a more optimistic one.
        '''

        test_obs_seq = ''.join(map(str, self.test_obs_list))
        viterbi = Viterbi('yo.txt', self.matrices.trans_mat, self.matrices.emiss_mat, test_obs_seq)
        prediction_seq = viterbi.run(test_obs_seq)
        test_state_seq = ''.join(map(str, self.test_state_list))

        mismatch_count = 0
        for j in range(0, len(prediction_seq)):
            if prediction_seq[j] != test_state_seq[j]:
                mismatch_count += 1
        error = mismatch_count / float(len(prediction_seq))
        # If this if statement is not here, we will not pocket the minimum error,
        # and will instead just use the error at the end of gradient descent (a more
        # realistic view)
        #if error < self.validation_errors[self.cross_val_slice]:
        self.validation_errors[self.cross_val_slice] = error
        return error

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
        self.state_seq = ''.join(map(str, self.state_list))
        self.test_state_list = old_state_list[start:start + size]

        old_obs_list = self.obs_list
        self.obs_list = self.obs_list[0:start] + self.obs_list[start + size:]
        self.obs_seq = ''.join(map(str, self.obs_list))
        self.test_obs_list = old_obs_list[start:start + size]

    def reset_data(self):
        ''' Resets data to original file inputs.'''

        self.pair_list = self.list_dict['pair_list']
        self.state_list = self.list_dict['state_list']
        self.state_seq = ''.join(map(str, self.state_list))
        self.obs_list = self.list_dict['obs_list']
        self.obs_seq = ''.join(map(str, self.obs_list))

class Matrices:
    def __init__(self, num_states, num_observations, obs_seq):
        '''Constructor for the Matrices class. This class stores the
        transition and emission matrices used for CRF, Forward, and Backward
        algorithms. It also has all the methods for performing updates to these
        matrices.
        '''

        self.num_states = num_states
        self.num_observations = num_observations
        self.emiss_mat = []
        self.trans_mat = []
        self.obs_seq = obs_seq

        self.init_trans_mat()
        self.init_emiss_mat()
        self.init_shift_trans_mat()
        self.init_shift_emiss_mat()
        # The learning rate for gradient descent
        self.learning_rate = .0001
        self.stopping_point = .001

    def init_trans_mat(self):
        '''Makes an initial transition matrix, where the values are chosen from
        a uniform distribution between 0 and 1. Our transition
        matrix is defined as follows. An element (a, b) of the matrix represents
        the score of transitioning from state b to state a.
        '''

        self.trans_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, self.num_states):
                row.append(random.random())
            self.trans_mat.append(row)

    def init_emiss_mat(self):
        '''Makes an initial emission matrix, where the values are chosen from
        a uniform distribution between 0 and 1. Our emission
        matrix is defined as follows. An element (a, b) of the matrix represents
        the score of having an observation b given state a.
        '''

        self.emiss_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, self.num_observations):
                row.append(random.random())
            self.emiss_mat.append(row)

    def init_shift_trans_mat(self):
        '''Makes an initial shift matrix for the transition matrix. This matrix
        is used for gradient descent. After every epoch, we add this matrix to
        the transition matrix to do our gradient step.
        '''

        self.shift_trans_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, self.num_states):
                row.append(0)
            self.shift_trans_mat.append(row)

    def init_shift_emiss_mat(self):
        '''Makes an initial shift matrix for the emission matrix. This matrix
        is used for gradient descent. After every epoch, we add this matrix to
        the emission matrix to do our gradient step.
        '''

        self.shift_emiss_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, self.num_observations):
                row.append(0)
            self.shift_emiss_mat.append(row)

    def get_g_score_forward(self, seq_pos, state_curr, state_prev, seq_int):
        '''Returns the G-score for the forward algorithm. This is just
        e^{trans_score + emiss_score}.
        '''

        trans_score = self.trans_mat[state_curr][state_prev]
        emiss_score = self.emiss_mat[state_curr][seq_int]
        g_score = math.exp(trans_score + emiss_score)
        return g_score

    def get_g_score_backward(self, seq_pos, state_curr, state_next, seq_int):
        '''Returns the G-score for the backward algorithm. It's basically
        the same as the forward version, but with different parameter names
        to give more clarity.
        '''

        trans_score = self.trans_mat[state_next][state_curr]
        emiss_score = self.emiss_mat[state_next][seq_int]
        g_score = math.exp(trans_score + emiss_score)
        return g_score

    def update_shift_trans_mat(self, state_curr, state_prev, val):
        '''Updates the shift transition matrix at a specific cell, using
        the learning rate of the class since the update is for gradient descent.
        '''

        self.shift_trans_mat[state_curr][state_prev] -= self.learning_rate * val

    def update_shift_emiss_mat(self, state_curr, seq_pos, val):
        '''Updates the shift emission matrix at a specific cell, using
        the learning rate of the class since the update is for gradient descent.
        '''

        seq_int = int(self.obs_seq[seq_pos])
        self.shift_emiss_mat[state_curr][seq_int] -= self.learning_rate * val

    def update_trans_mat(self):
        '''Updates the transition matrix at every cell by applying the shift transition
        matrix.
        '''

        #print 'Updating trans_mat by the following shift matrix:'
        #print self.shift_trans_mat
        #print 'Old trans_mat is:'
        #print self.trans_mat
        self.trans_mat = [map(sum, zip(*t)) for t in zip(self.trans_mat, self.shift_trans_mat)]
        print 'New trans_mat is:'
        print self.trans_mat
        #print '\n'

    def update_emiss_mat(self):
        '''Updates the emission matrix at every cell by applying the shift emission
        matrix.
        '''

        #print 'Updating emiss_mat by the following shift matrix:'
        #print self.shift_emiss_mat
        #print 'Old emiss_mat is:'
        #print self.emiss_mat
        self.emiss_mat = [map(sum, zip(*t)) for t in zip(self.emiss_mat, self.shift_emiss_mat)]
        print 'New emiss_mat is:'
        print self.emiss_mat

    def shift_has_nan(self):
        '''Checks to see if either of the shift matrices has a nan value.
        This method is used to terminate gradient descent.
        '''

        for row in self.shift_trans_mat:
            for val in row:
                if np.isnan(val):
                    return True
        for row in self.shift_emiss_mat:
            for val in row:
                if np.isnan(val):
                    return True
        return False

    def shift_has_inf(self):
        '''Checks to see if either of the shift matrices has an inf value.
        This method is used to terminate gradient descent.
        '''

        for row in self.shift_trans_mat:
            for val in row:
                if np.isinf(val):
                    return True
        for row in self.shift_emiss_mat:
            for val in row:
                if np.isinf(val):
                    return True
        return False

    def is_changing(self):
        '''Returns true is either the trans mat or emiss mat is still changing
        by a value specified in the constructor and false otherwise.
        '''

        if self.trans_mat_is_changing() or self.emiss_mat_is_changing():
            return True
        return False

    def trans_mat_is_changing(self):
        '''Returns true if the trans mat is still changing by a value
        specified in the constructor and false otherwise.
        '''

        for row in self.shift_trans_mat:
            for val in row:
                if val > self.stopping_point:
                    return True
        return False

    def emiss_mat_is_changing(self):
        '''Returns true if the emiss mat is still changing by a value
        specified in the constructor and false otherwise.
        '''

        for row in self.shift_emiss_mat:
            for val in row:
                if val > self.stopping_point:
                    return True
        return False

class Forward:
    def __init__(self, num_states, matrices):
        '''Constructor for the Forward class. This class runs the Forward
        algorithm for CRFs and saves the resulting DP matrix as an
        instance variable.
        '''

        self.matrices = matrices
        self.num_states = num_states
        self.forward_mat = []
        self.normalization_factors = []

    def run(self, sequence):
        '''Runs Foward on the passed in sequence.'''

        self.normalization_factors = [40 for x in range(len(sequence))]
        self.initialize_forward_mat(sequence)
        # Traverse by column
        for seq_pos in range(1, len(sequence)):
            # Compute dynamic multiplicative factors to prevent underflow/overflow
            normalization_factor = 0
            for state_curr in range(0, self.num_states):
                for state_prev in range(0, self.num_states):
                    normalization_factor += self.forward_mat[state_prev][seq_pos - 1] * self.matrices.get_g_score_forward(seq_pos, state_curr, state_prev, int(sequence[seq_pos]))
            self.normalization_factors[seq_pos] = normalization_factor

            for state_curr in range(0, self.num_states):
                seq_int = int(sequence[seq_pos])
                sum_score = 0
                for state_prev in range(0, self.num_states):
                    # Calculate score coming from each previous state/seq_pos and sum
                    # them up
                    forward_score = self.forward_mat[state_prev][seq_pos - 1]
                    g_score = self.matrices.get_g_score_forward(seq_pos, state_curr, state_prev, seq_int)
                    normalization_factor = self.normalization_factors[seq_pos]
                    score = float(g_score * forward_score) / normalization_factor
                    sum_score += score
                self.forward_mat[state_curr][seq_pos] = sum_score

    def get_val(self, state_index, col_index):
        ans = self.forward_mat[state_index][col_index]
        if print_forward:
            print 'Forward =', ans
        return ans

    def initialize_forward_mat(self, sequence):
        '''Initializes the forward matrix.'''

        self.forward_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, len(sequence)):
                if j == 0:
                    seq_int = int(sequence[j])
                    # Calculate initial probabilities.
                    emiss_val = self.matrices.emiss_mat[i][seq_int]
                    normalization_val = self.normalization_factors[j]
                    val = float(math.exp((1.0 / self.num_states) + emiss_val)) / normalization_val
                    row.append(val)
                else:
                    row.append(0)
            self.forward_mat.append(row)

    def print_mat(self):
        '''Prints the forward matrix.'''

        print '\n\n'
        for row in self.forward_mat:
            print row
            print '\n'

    def print_col(self, col_index):
        '''Prints the passed in column of the forward matrix.'''

        print 'Printing column', col_index, 'of forward_mat'
        for row_index in range(0, self.num_states):
            print self.forward_mat[row_index][col_index]

class Backward:
    def __init__(self, num_states, matrices):
        '''Constructor for the Backward class. This class runs the Backward
        algorithm for CRFs and saves the resulting DP matrix as an
        instance variable.
        '''

        self.matrices = matrices
        self.num_states = num_states
        self.backward_mat = []
        self.normalization_factors = []

    def run(self, sequence):
        '''Runs Backward on the passed in sequence.'''

        self.normalization_factors = [40 for x in range(len(sequence))]
        self.initialize_backward_mat(sequence)
        # Traverse by column, starting from second to last column
        for seq_pos in range(len(sequence) - 2, -1, -1):
            # Compute dynamic multiplicative factors to prevent underflow/overflow
            normalization_factor = 0
            for state_curr in range(0, self.num_states):
                for state_prev in range(0, self.num_states):
                    normalization_factor += self.backward_mat[state_curr][seq_pos + 1] * self.matrices.get_g_score_forward(seq_pos, state_curr, state_prev, int(sequence[seq_pos]))
            self.normalization_factors[seq_pos] = normalization_factor

            for state_curr in range(0, self.num_states):
                seq_int = int(sequence[seq_pos])
                sum_score = 0
                for state_next in range(0, self.num_states):
                    # Calculate score coming from each next state/seq_pos and sum
                    # them up.
                    # Note that we sum the emission score, as opposed to forward.
                    g_score = self.matrices.get_g_score_backward(seq_pos, state_curr, state_next, seq_int)
                    backward_score = self.backward_mat[state_next][seq_pos + 1]
                    normalization_factor = self.normalization_factors[seq_pos]
                    score = float(g_score * backward_score) / normalization_factor
                    sum_score += score
                self.backward_mat[state_curr][seq_pos] = sum_score

    def get_val(self, state_index, col_index):
        ans = self.backward_mat[state_index][col_index]
        if print_backward:
            print 'Backward =', ans
        return ans

    def initialize_backward_mat(self, sequence):
        '''Runs Backward on the passed in sequence.'''

        self.backward_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, len(sequence)):
                if j == len(sequence) - 1:
                    # For backward, initial val is 1
                    val = float(1) / self.normalization_factors[j]
                    row.append(val)
                else:
                    row.append(0)
            self.backward_mat.append(row)

    def print_mat(self):
        '''Prints the backward matrix.'''

        print '\n\n'
        for row in self.backward_mat:
            print row
            print '\n'

    def print_col(self, col_index):
        '''Prints the passed in column of the forward matrix.'''

        print 'Printing column', col_index, 'of backward_mat'
        for row_index in range(0, self.num_states):
            print self.backward_mat[row_index][col_index]

if __name__ == '__main__':
    crf = CRF('ron.txt')
    #crf.run_gradient_descent()
    crf.cross_validate()
