from logging import root
import numpy as np
import pandas as pd
from copy import  deepcopy
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class Node:
    def __init__(self, root: str) -> None:
        self.root= root
        self.nodes = dict()
        self.end = False
        self.answer = ""
        self.rules = []
        self.current_rule = []
    
    def add_child(self, child, option):
        self.nodes[option] = child
    
    def add_answer(self, answer):
        self.end = True
        self.answer = answer
    
    def get_child(self, child):
        return self.nodes[child]
    
    def get_children(self):
        return self.nodes.keys()
    
    def get_root(self):
        return self.root

    def has_children(self):
        return len(self.nodes) != 0
    
    def get_rules(self, Node, OG_NODE):
        if isinstance(Node, str):
            OG_NODE.current_rule = [OG_NODE.current_rule]
            OG_NODE.current_rule.append(Node)
            OG_NODE.rules.append(tuple(deepcopy(OG_NODE.current_rule)))
            OG_NODE.current_rule.pop()
            OG_NODE.current_rule = OG_NODE.current_rule[0]
            return
        OG_NODE.current_rule.append([Node.root])
        for child in Node.nodes.keys():
            OG_NODE.current_rule[-1].append(child)
            OG_NODE.get_rules(Node.nodes[child], OG_NODE)
            OG_NODE.current_rule[-1].pop()
        OG_NODE.current_rule.pop()
        return



class DecisionTree:

    def __init__(self):
        self.rules: Node
        self.attributes = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement
        self.attributes = X.columns
        x_entropy = entropy(pd.Series([len(y[y == elem]) for elem in set(y)]))
        if x_entropy == 0:
            return y.values[0]
        gainz = dict()
        for attribute in self.attributes:
            gainz[attribute] = self._gain(attribute, set(X[attribute]), x_entropy, X, y)
        
        if(len(gainz) == 0): return y.value_counts().idxmax()
        best_attribute = max(gainz, key=gainz.get)
        
        
        tree_node = Node(best_attribute)
        tree_node.answer = y.value_counts().idxmax()
        for attribute_value in set(X[best_attribute]):
            new_X = X.query(f"`{best_attribute}` == '{attribute_value}'")
            value = self.fit(X = new_X.drop(columns=[best_attribute]),y= y[new_X.index])
            if isinstance(value, str): 
                tree_node.answer = value
                pass
            tree_node.add_child(value, attribute_value)
        self.rules = tree_node
        return tree_node
        


    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        output_vector = []
        for index, row in X.iterrows():
            predictor = self.rules
            while predictor.end != True:
                if row[predictor.get_root()] not in predictor.get_children(): #If not in training set
                    predictor = predictor.answer
                    break
                predictor = predictor.get_child(row[predictor.get_root()])
                if isinstance(predictor, str): break
            output_vector.append(predictor) 
        output_vector = pd.Series(output_vector)
        output_vector.index += X.index[0]
        return output_vector
            


    def get_rules(self):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label

            attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        self.rules.rules = []
        self.rules.get_rules(self.rules, self.rules)
        return self.rules.rules

        # TODO: Implement
         

    def _gain(self, attribute, attributes, ent, X: pd.DataFrame, y: pd.DataFrame):
        """return ent - [(len(X.query(f"{attribute} == '{att_type}'")) /
                      len(X)) *
                      entropy([len(y[X.query(f"{attribute} == '{att_type}'").index].str.find(elem))
                    for elem in set(y[X.query(f"{attribute} == '{att_type}'").index])]) 
                    for att_type in attributes].sum()"""
        #Prøvde å gjøre det på en linje, ga opp
        #Fant ut at det var grunnet dårlig dokumentasjon i entropy-funksjonen >:(
        #MAN TRENGER MER ENN "ARRAY(<k>)??? Man trenger enten numpy eller pandas >>>>:(((("
        entrop = 0
        for att_type in attributes:
            X_sv = X.query(f"`{attribute}` == '{att_type}'")
            y_sv = y[X_sv.index]
            entropy_sv = [len(y_sv[y_sv == elem]) for elem in set(y)]
            entrop += y_sv.size/y.size * entropy(pd.Series(entropy_sv))
        return ent - entrop
            


# --- Some utility functions

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))
