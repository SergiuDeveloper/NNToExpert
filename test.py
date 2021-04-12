import tensorflow as tf
from NNToExpert import *


nn = tf.keras.models.load_model('models/xor')

variable_domains = [Variable(DiscreteDomain([0, 1])), Variable(DiscreteDomain([0, 1]))]

expert = None
with open('test.clp', 'w+') as output_file:
    expert = NNToExpert.extract_rules(nn, variable_domains, output_file)

for v0 in range(2):
    for v1 in range(2):
        result = expert.classify([v0, v1])
        print('f({}, {}) = {}'.format(v0, v1, result))