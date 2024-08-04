"""
(Word-level Textual Adversarial Attacking as Combinatorial Optimization)
"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import Rubbishexample
from textattack.transformations import WordSwapStopwords
from textattack.search_methods import MOSA

from .attack_recipe import AttackRecipe


class MOSAS2Li2023(AttackRecipe):
    """

        Multi-Objective Simulated Annealing based Stopwords Substitution for Rubbish Text Attack

    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with stopwords.
        #
        transformation = WordSwapStopwords()
        #transformation = WordDeletion()

        #
        # Don't modify the same word twice
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        #
        # During entailment, we should only edit the hypothesis - keep the premise
        # the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        #
        # The goal is to maximize the confidence of the original label and word modifcation rate.
        #
        goal_function = Rubbishexample(model_wrapper,maximizable=True)
        #
        # Perform word substitution with an multi-objective simulated annealing algorithm.
        #
        search_method = MOSA(wir_method="gradient")
        return Attack(goal_function, constraints, transformation, search_method)
