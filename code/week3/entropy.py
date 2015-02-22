import math

p = raw_input('Enter p: ')
g = raw_input('Enter g for gini index: ')
p = float(p)
if not g == 'g':
    if p == 0:
        first_term = 0
    else:
        first_term = p * math.log(p, 2)
    if p == 1:
        second_term = 0
    else:
        second_term = (1 - p) * math.log(1 - p, 2)
    entropy = -(first_term + second_term)
else:
    print 'Using gini index'
    entropy = 1 - (math.pow(p, 2) + math.pow((1 - p), 2))
print 'p =', p
print 'Entropy =', entropy
