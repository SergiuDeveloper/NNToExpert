import tensorflow as tf
import numpy as np
from clips import Environment
from rule_extraction import *


X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([
    [0],
    [0],
    [0],
    [1]
])

nn = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
])
nn.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
)
nn.fit(X_train, y_train, epochs=30, batch_size=1, verbose=False)

variable_domains = [Variable(DiscreteDomain([0, 1])), Variable(DiscreteDomain([0, 1]))]
rules_text = NNToExpertHelper.extract_rules(nn, variable_domains)
with open('test.clp', 'w+') as output_file:
    output_file.write('\n'.join(rules_text))

clips_env = Environment()
for rule_text in rules_text:
    clips_env.build(rule_text)
for v0 in range(2):
    for v1 in range(2):
        clips_env.assert_string('(input_vars {} {})'.format(v0, v1))
        clips_env.run()
        for fact in clips_env.facts():
            fact_text = str(fact)
            if fact_text.startswith('(output_vars'):
                output_vars_str = fact_text[len('(output_vars '):-1]
                print('f({}, {}) = {}'.format(v0, v1, output_vars_str))
        clips_env.reset()