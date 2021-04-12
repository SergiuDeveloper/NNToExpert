import itertools
from abc import ABC
from typing import List
import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, _tree


class ContinuousSubdomain():
    def __init__(self, domain_low: float, domain_high: float, closed_low: bool, closed_high: bool, step: float):
        self.values = np.arange(domain_low + (step if not closed_low else 0), domain_high - (step if not closed_high else 0), step)
        
class Domain(ABC):
    def __init__(self, values: List[float]):
        self.values: List[float] = sorted(values)
        
class DiscreteDomain(Domain):
    def __init__(self, values: List[float]):
        super().__init__(values)

class ContinuousDomain(Domain):
    def __init__(self, subdomains: List[ContinuousSubdomain]):
        subdomains_values = [subdomain.values for subdomain in subdomains]
        values = list(itertools.chain.fromiterable(subdomains_values))
        super().__init__(values)

class Variable():
    def __init__(self, domain: Domain):
        self.domain: Domain = domain
        
class NNToExpert():
    @staticmethod
    def extract_rules(nn: tf.keras.Model, variables: List[Variable]) -> List[str]:
        NNToExpert.rules_count = 0
        NNToExpert.rules_text = []
    
        variable_names = ['V{}'.format(i) for i in range(len(variables))]
        variables_domain_values = [variable.domain.values for variable in variables]
        variables_domains_cartesian_product = list(itertools.product(*variables_domain_values))
        predicted_values = nn.predict(variables_domains_cartesian_product)
        predicted_values = [int(predicted_value[0] > 0.5) for predicted_value in predicted_values]
        
        clf = DecisionTreeClassifier()
        clf.fit(variables_domains_cartesian_product, predicted_values)
        NNToExpert.__tree_to_rules(clf, variable_names)
        
        return NNToExpert.rules_text
        
    @staticmethod
    def __tree_to_rules(clf, feature_names):
        feature_names = [feature_names[i] if i != _tree.TREE_UNDEFINED else 'undefined' for i in clf.tree_.feature]
        NNToExpert.__tree_to_rules_aux(clf, 0, 1, feature_names, '')
            
    @staticmethod
    def __tree_to_rules_aux(clf, node, depth, feature_names, current_rules_text):
        if clf.tree_.feature[node] == _tree.TREE_UNDEFINED:
            leaf_value = np.argmax(clf.tree_.value[node])

            clips_input_vars = ['?{}'.format(feature_name) for feature_name in feature_names if feature_name != 'undefined']
            input_line = '(input_vars {})'.format(' '.join(map(str, sorted(clips_input_vars))))
            rule_text = '(defrule r{}\n\t{}\n{}=>\n\t(assert (output_vars {}))\n)\n'.format(NNToExpert.rules_count, input_line, current_rules_text, leaf_value)
            NNToExpert.rules_text.append(rule_text)
                
            NNToExpert.rules_count += 1
            return
            
        left_rules_text = '{}\t(test (<= ?{} {}))\n'.format(current_rules_text, feature_names[node], clf.tree_.threshold[node])
        NNToExpert.__tree_to_rules_aux(clf, clf.tree_.children_left[node], depth + 1, feature_names, left_rules_text)
            
        right_rules_text = '{}\t(test (> ?{} {}))\n'.format(current_rules_text, feature_names[node], clf.tree_.threshold[node])
        NNToExpert.__tree_to_rules_aux(clf, clf.tree_.children_right[node], depth + 1, feature_names, right_rules_text)
