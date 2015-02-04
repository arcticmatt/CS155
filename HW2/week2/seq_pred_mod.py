import sys
import numpy as np

# THIS IS EXACTLY THE SAME AS seq_pred.py, EXCEPT FOR LINE 88 (WHERE WE ADD INSTEAD OF MULTIPLY)
class HMM:
    def __init__(self, filename):
        '''Constructor for the HMM class. Basically just a convenience class
        that loads the file for question 1.
        '''

        self.num_observations = 0
        self.num_states = 0
        self.state_trans_mat = []
        self.output_emiss_mat = []
        self.sequences = []
        self.load_hmm(filename)

    def load_hmm(self, filename):
        '''Loads the HMM file that the TAs give us. The first row
        contains two tab-delimited numbers; the number of states Y and the
        number of types of observations X. The X observations emit outputs
        0, 1, ..., X - 1. The next Y rows of Y tab-delimited floating-point numbers
        describe the state transition matrix. The next Y rows of tab-delimited
        floating-point numbers describe the output emission matrix, encoded
        analgously to the state transition matrix. The file ends with 5 possible
        emissions from that HMM.
        '''

        file_obj = open(filename)
        file_info = file_obj.readline().split()
        self.num_states = int(file_info[0])
        self.num_observations = int(file_info[1])
        for i in range (0, self.num_states):
            row = map(float, file_obj.readline().split())
            self.state_trans_mat.append(row)
        for i in range (0, self.num_states):
            row = map(float, file_obj.readline().split())
            self.output_emiss_mat.append(row)
        for remaining_line in file_obj.readlines():
            self.sequences.append(remaining_line.strip())

class Viterbi(HMM):
    def __init__(self, filename, trans_mat = None, emiss_mat = None, sequence = None):
        '''Constructor for the Viterbi class. This class runs the Viterbi
        algorithm on one or several sequences.
        '''

        if trans_mat != None and emiss_mat != None and sequence != None:
            self.state_trans_mat = trans_mat
            self.output_emiss_mat = emiss_mat
            self.sequences = [sequence]
            self.num_observations = len(emiss_mat[0])
            self.num_states = len(trans_mat)
        else:
            HMM.__init__(self, filename)
            print 'here'
        self.viterbi_mat = []

    def run_all(self):
        '''Runs Viterbi on all sequences of the class.'''

        for sequence in self.sequences:
            self.run(sequence)

    def run(self, sequence):
        '''Runs Viterbi on the passed in sequence. Returns the maximum
        probability hidden state sequence.
        '''

        self.initialize_viterbi_mat(sequence)
        # Traverse by column
        for i in range(1, len(sequence)):
            for j in range(0, self.num_states):
                # Get the max score (probability )for each observation, along with
                # the state which generated that score. We will use that state
                # for backtracking.
                seq_int = int(sequence[i])
                emiss_score = self.output_emiss_mat[j][seq_int]
                max_score = -sys.maxint - 1
                old_index = 0
                for k in range(0, self.num_states):
                    # Calculate score coming from each previous state. In
                    # the outer-loop we take the max.
                    trans_score = self.state_trans_mat[k][j]
                    viterb_score = self.viterbi_mat[k][i-1][0]
                    # Note that when using Viterbi for CRF, we want to add these
                    # scores instead of multiplying them
                    score = np.longdouble(emiss_score + trans_score + viterb_score)
                    if score > max_score:
                        max_score = score
                        old_index = k
                self.viterbi_mat[j][i] = (max_score, old_index)

        # Find max value in last column.
        max_prob = -sys.maxint - 1
        column = 0
        for i in range(0, self.num_states):
            prob = self.viterbi_mat[i][len(sequence) - 1]
            if prob > max_prob:
                max_prob = prob
                column = i

        # Backtrack from last tuple.
        answer = []
        answer.insert(0, column) # add the state for the last column
        # Add the states from all other columns. Note that we do not reach
        # column 0; this is because column 1 points to the state that we
        # want for column 0.
        for i in range(len(sequence) - 1, 0, -1):
            viterb_cell = self.viterbi_mat[column][i]
            column = viterb_cell[1]
            answer.insert(0, column)

        # Turn answer list into a string for printing
        answer_str = ''.join(map(str, answer))
        print 'MAXSTATESEQ = ' + answer_str + ' \\\\'
        return answer_str

    def initialize_viterbi_mat(self, sequence):
        '''Method to initialize our DP matrix. Each cell of the matrix stores a
        tuple. The first index of the tuple holds the probability of the
        cell. If we are looking at row R, column C, this probability is the max
        probability of having R-lengh sequence ending in the tag represented by
        column C. The second index of the tuple stores the value of the row of the last
        column that gave the cell its optimal probability (used for backtracking).
        In other words, it points back to the previous cell.
        '''

        self.viterbi_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, len(sequence)):
                if j == 0:
                    seq_int = int(sequence[0])
                    # Calculate initial probabilities.
                    #val = (1.0 / self.num_states) * self.output_emiss_mat[i][seq_int]
                    val = (1.0 / self.num_states) * self.output_emiss_mat[i][seq_int]
                    row.append((val, 0))
                else:
                    row.append((0, 0))
            self.viterbi_mat.append(row)

    def print_viterbi_mat(self):
        '''Prints the viterbi matrix.'''

        print '\n\n'
        for row in self.viterbi_mat:
            print row
            print '\n'

class Forward(HMM):
    def __init__(self, filename):
        '''Constructor for the Forward class. This class runs the regular Forward
        algorithm and saves the resulting DP matrix as an instance variable.
        By "regular" Forward algorithm, I am referring to the Forward algorithm
        that uses transition and emission matrices with probabilities, as opposed
        to the Forward algorithm used for CRF.
        '''
        HMM.__init__(self, filename)
        self.forward_mat = []

    def run_all(self):
        '''Runs Forward on all sequences of the class.'''

        for sequence in self.sequences:
            self.run(sequence)

    def run(self, sequence):
        '''Runs Foward on the passed in sequence, and returns the probability
        of getting that sequence.
        '''

        self.initialize_forward_mat(sequence)
        # Traverse by column
        for seq_pos in range(1, len(sequence)):
            for state_curr in range(0, self.num_states):
                seq_int = int(sequence[seq_pos])
                emiss_score = self.output_emiss_mat[state_curr][seq_int]
                sum_score = 0
                for state_prev in range(0, self.num_states):
                    # Calculate score coming from each previous state and sum
                    # them up
                    trans_score = self.state_trans_mat[state_prev][state_curr]
                    forward_score = self.forward_mat[state_prev][seq_pos - 1]
                    score = emiss_score * trans_score * forward_score
                    sum_score += score
                self.forward_mat[state_curr][seq_pos] = sum_score

        ans = 0
        # Sum up probabilities in last column (these are the probabilities
        # that we get our sequence ending in different states)
        for i in range(0, self.num_states):
            ans += self.forward_mat[i][len(sequence) - 1]
        print 'Forward:'
        print 'P(' + sequence + ') =', ans, ' \\\\'
        return ans

    def initialize_forward_mat(self, sequence):
        '''Initializes the forward matrix.'''

        self.forward_mat = []
        for i in range(0, self.num_states):
            row = []
            for j in range(0, len(sequence)):
                if j == 0:
                    seq_int = int(sequence[0])
                    # Calculate initial probabilities.
                    val = (1.0 / self.num_states) * self.output_emiss_mat[i][seq_int]
                    row.append(val)
                else:
                    row.append(0)
            self.forward_mat.append(row)

    def print_forward_mat(self):
        '''Prints the forward matrix.'''

        print '\n\n'
        for row in self.forward_mat:
            print row
            print '\n'

if __name__ == '__main__':
    file_list = ['sequenceprediction1.txt', 'sequenceprediction2.txt',
            'sequenceprediction3.txt', 'sequenceprediction4.txt',
            'sequenceprediction5.txt']
    for filename in file_list:
        #print '==================== FILE = ', filename, '===================='
        print 'FILE = ', filename
        #print '---------- VITERBI ----------'
        viterbi = Viterbi(filename)
        viterbi.run_all()

        #print '\n'
        #print '---------- FORWARD ----------'
        forward = Forward(filename)
        forward.run_all()
        print '\n'
