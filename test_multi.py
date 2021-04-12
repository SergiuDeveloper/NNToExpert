import tensorflow as tf
import numpy as np
import itertools
from NNToExpert import *


nn = tf.keras.models.load_model('models/multi')

variables = [
    Variable(ContinuousDomain([ContinuousSubdomain(0, 1, True, True, 0.1)]))
    for i in range(3)
]

with open('test.clp', 'w+') as output_file:
    expert = NNToExpert.extract_rules(nn, variables, output_file)

results = []
expected_values = []
variable_domains_values = [variable.domain.values for variable in variables]
cartesian_product = np.array(list(itertools.product(*variable_domains_values)))

expected_values = [int(round(result[0])) for result in nn.predict(cartesian_product)]
results = expert.classify(cartesian_product)

assert(expected_values == results)

for i in range(len(cartesian_product)):
    print('f({}) = {}'.format(', '.join(map(str, cartesian_product[i])), results[i]))