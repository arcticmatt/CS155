from sklearn import tree
import csv
import matplotlib.pyplot as plt

class DTree:
    def __init__(self, filename):
        self.min_samples_leaves = [i for i in range(1, 26)]
        self.max_depths = [i for i in range(2, 21)]
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.test_errors = []
        self.train_errors = []
        self.load_file(filename)
        print self.Y_train

    def load_file(self, filename):
        file_obj = open(filename)
        count = 0
        for line in file_obj.readlines():
            data = line.strip().split(',')
            # Ignore first column which is just ID number
            x = map(float, data[2:])
            y = data[1]
            if count < 400:
                self.X_train.append(x)
                self.Y_train.append(y)
            else:
                self.X_test.append(x)
                self.Y_test.append(y)
            count += 1

    def run_max_depths(self):
        self.test_errors = []
        self.train_errors = []
        for max_depth in self.max_depths:
            # Initialize the tree model
            clf = tree.DecisionTreeClassifier(criterion='gini',
                    max_depth=max_depth)

            self.run(clf)
        self.draw_plot(self.max_depths, 'max_depth', 'Plot of Error vs. max_depth', 'max_depths.png')

    def run_min_samples_leaves(self):
        self.test_errors = []
        self.train_errors = []
        for min_samples_leaf in self.min_samples_leaves:
            # Initialize the tree model
            clf = tree.DecisionTreeClassifier(criterion='gini',
                    min_samples_leaf=min_samples_leaf)

            self.run(clf)
        self.draw_plot(self.min_samples_leaves, 'min_samples_leaf', 'Plot of Error vs. min_samples_leaf', 'min_samples_leaves.png')

    def run(self, clf):
        # Train the model
        clf = clf.fit(self.X_train, self.Y_train)

        # Make prediction
        G_train = clf.predict(self.X_train)
        G_test = clf.predict(self.X_test)

        # Compute error
        train_error = self.get_error(G_train, self.Y_train)
        self.train_errors.append(train_error)
        test_error = self.get_error(G_test, self.Y_test)
        self.test_errors.append(test_error)


    def draw_plot(self, x_data, x_label, title, filename):
        # Draw the plot
        plt.clf()
        plt.plot(x_data, self.train_errors)
        plt.plot(x_data, self.test_errors)
        plt.xlabel(x_label)
        plt.ylabel('Error')
        plt.title(title)
        plt.legend(['train_error', 'test_error'])
        # plt.show()
        save_dir = '../../psets/pset3/images/' + filename
        plt.savefig(save_dir, bbox_inches='tight')

    def get_error(self, G, Y):
        error = 0
        for i in range(len(G)):
            if G[i] != Y[i]:
                error += 1
        return 1.0 * error / len(G)

if __name__ == '__main__':
    dtree = DTree('wdbc.data')
    dtree.run_min_samples_leaves()
    dtree.run_max_depths()
