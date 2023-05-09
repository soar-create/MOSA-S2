"""

Particle Swarm Optimization
==================================

(Word-level Textual Adversarial Attacking as Combinatorial Optimization)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import Rubbishexample
from textattack.transformations import WordSwapPreposition
from textattack.search_methods import AnnealingGeneticAlgorithm
from .attack_recipe import AttackRecipe


class AGPS2023(AttackRecipe):
    """

        Annealing Genetic based Preposition Substitution for Text Rubbish Example Generation

    """

    @staticmethod
    def build(model_wrapper):
        #
        # Swap words with prepositions.
        #
        transformation = WordSwapPreposition()
        #
        # Don't modify the same word twice or stopwords
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
        goal_function = Rubbishexample(model_wrapper, maximizable=True)
        #
        # Perform word substitution with an Annealing Genetic algorithm.
        #
        search_method = AnnealingGeneticAlgorithm(pop_size=38, iters=12, post_crossover_check=False)
        return Attack(goal_function, constraints, transformation, search_method)
