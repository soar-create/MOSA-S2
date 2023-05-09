"""
Annealing Genetic Algorithm Preposition Swap
====================================
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import math

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import PopulationBasedSearch, PopulationMember
from textattack.shared.validators import transformation_consists_of_word_swaps


class AnnealingGeneticAlgorithm(PopulationBasedSearch, ABC):
    """An attack that select the best rubbish sample from a list of possible preposition substiutitions 
    using annealing genetic algorithm.

    Args:
        pop_size (int): The population size. Defaults to 40.
        iters (int): The number of evolutionary iterations. Defaults to 15.
        give_up_if_no_improvement (bool): If True, stop the search early if no candidate that improves the score is found.
        post_crossover_check (bool): If True, check if child produced from crossover step passes the constraints.
        max_crossover_retries (int): Maximum number of crossover retries if resulting child fails to pass the constraints.
            Applied only when `post_crossover_check` is set to `True`.
            Setting it to 0 means we immediately take one of the parents at random as the child upon failure.
    """

    def __init__(
        self,
        pop_size=40,
        iters=15,
        give_up_if_no_improvement=False,
        post_crossover_check=True,
        max_crossover_retries=20,
    ):
        self.iters = iters
        self.pop_size = pop_size
        self.give_up_if_no_improvement = give_up_if_no_improvement
        self.post_crossover_check = post_crossover_check
        self.max_crossover_retries = max_crossover_retries

        # internal flag to indicate if search should end immediately
        self._search_over = False

    
    def crossover_operation(self, pop_member1, pop_member2, T):
        """Actual operation that mixes `pop_member1` text and `pop_member2`
        text to generate crossover.

        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
        Returns:
            Tuple of `AttackedText` and a dictionary of attributes.
        """
        indices_to_replace = []
        words_to_replace = []
        pop_list=[]
        pop_list.append(pop_member1.attacked_text)
        pop_list.append(pop_member2.attacked_text)
        num_candidate_transformations = np.copy(
            pop_member1.attributes["num_candidate_transformations"]
        )
        # Exchanges words with the corresponding position and only modifies the input words with semantic meanings.
        for i in range(pop_member1.num_words):
            word_pos = pop_member1.attacked_text.pos_of_word_index(i)
            word_rep = pop_member1.attacked_text.words[i]
            pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}
            nonpos_word={"A":1,"An":1,"a":1,"an":1,"s":1,"is":1,"are":1,"been":1,"am":1,"be":1,"was":1,"were":1,"has":1,"have":1}
            word_pos = pos_dict.get(word_pos, None)
            word_nonpos = nonpos_word.get(word_rep, None)
            if word_pos!=None and word_nonpos == None:
                if np.random.uniform() < 0.7:
                    indices_to_replace.append(i)
                    words_to_replace.append(pop_member2.words[i])
                    num_candidate_transformations[i] = pop_member2.attributes[
                        "num_candidate_transformations"
                    ][i]

        cross_text = pop_member1.attacked_text.replace_words_at_indices(
            indices_to_replace, words_to_replace
        )
        pop_list.append(cross_text)
        pop_results, self._search_over = self.get_goal_results(
            pop_list
        )
        cross_score=pop_results[2].score
        pop_scores = np.array([r.score for r in pop_results])
        best_idx = np.argmax(pop_scores)
        y=pop_results[best_idx].score

        if pop_results[best_idx].attacked_text == cross_text:
            new_text=cross_text
        else:
            # metropolis principle
            p_cross=math.exp(-(y-cross_score)/T)
            if np.random.uniform() < p_cross and pop_results[2].score > 0:
                new_text=cross_text
            else:
                new_text=pop_results[best_idx].attacked_text
        return (
            new_text,
            {"num_candidate_transformations": num_candidate_transformations},
        )
    
    def mutation(self, pop_member, original_result, index=None):
        """ `pop_member` mutation and return it. Replaces a word at a random in `pop_member` 
        with the candidate substitution.
        Args:
            pop_member (PopulationMember): The population member being mutated.
            original_result (GoalFunctionResult): Result of original sample being attacked.
            index (int): Index of word being mutated.
        Returns:
            Mutated `PopulationMember`
        """
        num_words = pop_member.attacked_text.num_words
        # `word_select_prob_weights` is a list of values used for sampling one word to transform
        word_select_prob_weights = np.copy(
            self.get_word_select_prob_weights(pop_member)
        )
        non_zero_indices = np.count_nonzero(word_select_prob_weights)
        if non_zero_indices == 0:
            return pop_member
        iterations = 0
        while iterations < non_zero_indices:
            if index:
                idx = index
            else:
                w_select_probs = word_select_prob_weights / np.sum(
                    word_select_prob_weights
                )
                idx = np.random.choice(num_words, 1, p=w_select_probs)[0]
            word_pos = pop_member.attacked_text.pos_of_word_index(idx)
            word_rep = pop_member.attacked_text.words[idx]
            pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}
            nonpos_word={"A":1,"An":1,"a":1,"an":1,"s":1,"is":1,"are":1,"been":1,"am":1,"be":1,"was":1,"were":1,"has":1,"have":1}
            word_pos = pos_dict.get(word_pos, None)
            word_nonpos = nonpos_word.get(word_rep, None)
            global transformed_texts
            if word_pos!=None and word_nonpos == None:
                #print(word_pos)
                transformed_texts = self.get_transformations(
                    pop_member.attacked_text,
                    original_text=original_result.attacked_text,
                    indices_to_modify=[idx],
                )
                if not len(transformed_texts):
                    iterations += 1
                    continue
            new_results, self._search_over = self.get_goal_results(transformed_texts)
            diff_scores = (
                torch.Tensor([r.score for r in new_results]) - pop_member.result.score
            )
            if len(diff_scores):
                idx_with_max_score = diff_scores.argmax()
                pop_member = self.modify_population_member(
                    pop_member,
                    transformed_texts[idx_with_max_score],
                    new_results[idx_with_max_score],
                    idx,
                )
                return pop_member

            word_select_prob_weights[idx] = 0
            iterations += 1

            if self._search_over:
                break

        return pop_member

    def perform_search(self, initial_result):
        self._search_over = False
        population = self._initialize_population(initial_result, self.pop_size)
        pop_size = self.pop_size
        current_score = initial_result.score
        T = 1000 # initial temperature
        Ta=0.85 # the attenuation factor
        Tmin = (math.pow(Ta,self.iters))*T  # lowest temperature
        while T >= Tmin:
            #Sort population members for each generation by the fitness scores  
            population = sorted(population, key=lambda x: x.result.score, reverse=True)

            if self._search_over:
                break
            if population[0].result.score > current_score:
                current_score = population[0].result.score
                best_pop=population[0].result    # record the best individuals
            elif self.give_up_if_no_improvement:
                break
            # Well-performing population members have a higher probability of being selected as parents
            pop_scores = (torch.Tensor([pm.result.score for pm in population])).cpu().numpy()
            select_probs = pop_scores / np.sum(pop_scores)
            parent1_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(pop_size, size=pop_size - 1, p=select_probs)
            
            # Crossover
            children = []
            for idx in range(pop_size - 1):
                child = self.crossover(
                    population[parent1_idx[idx]],
                    population[parent2_idx[idx]],
                    T,
                    initial_result.attacked_text,
                )
                if self._search_over:
                    break
                # Mutation
                child_mut = self.mutation(child, initial_result)
                if child_mut.score > child.score:
                    child=child_mut
                else:
                    # Metropolis principle
                    p_mut=math.exp(-(child.score-child_mut.score)/T)
                    if np.random.uniform() < p_mut and child_mut.score > 0:
                        child=child_mut
                children.append(child)

                if self._search_over:
                    break

            population = [population[0]] + children # new generation population
            T=Ta*T  # annealing function
            # Return the optimal individual for the entire search
            if T<Tmin:
                if population[0].result.score < current_score:
                    population[0].result=best_pop
        return population[0].result

    def crossover(self, pop_member1, pop_member2, T, original_text):
        """Generates a crossover between pop_member1 and pop_member2.

        If the child fails to satisfy the constraints, we re-try crossover for a fix number of times,
        before taking one of the parents at random as the resulting child.
        Args:
            pop_member1 (PopulationMember): The first population member.
            pop_member2 (PopulationMember): The second population member.
            original_text (AttackedText): Original text
        Returns:
            A population member containing the crossover.
        """
        x1_text = pop_member1.attacked_text
        x2_text = pop_member2.attacked_text

        num_tries = 0
        passed_constraints = False
        while num_tries < self.max_crossover_retries + 1:
            new_text, attributes = self.crossover_operation(pop_member1, pop_member2, T)
            if "newly_modified_indices" in new_text.attack_attrs:
                replaced_indices = new_text.attack_attrs["newly_modified_indices"]
                new_text.attack_attrs["modified_indices"] = (
                    x1_text.attack_attrs["modified_indices"] - replaced_indices
                ) | (x2_text.attack_attrs["modified_indices"] & replaced_indices)

            if "last_transformation" in x1_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x1_text.attack_attrs[
                    "last_transformation"
                ]
            elif "last_transformation" in x2_text.attack_attrs:
                new_text.attack_attrs["last_transformation"] = x2_text.attack_attrs[
                    "last_transformation"
                ]

            if self.post_crossover_check:
                passed_constraints = self.post_crossover_check(
                    new_text, x1_text, x2_text, original_text
                )

            if not self.post_crossover_check or passed_constraints:
                break

            num_tries += 1

        if self.post_crossover_check and not passed_constraints:
            # If we cannot find a child that passes the constraints,
            # we just randomly pick one of the parents to be the child for the next iteration.
            pop_mem = pop_member1 if np.random.uniform() < 0.5 else pop_member2
            return pop_mem
        else:
            new_results, self._search_over = self.get_goal_results([new_text])
            return PopulationMember(
                new_text, result=new_results[0], attributes=attributes
            )

    def post_crossover_check(
        self, new_text, parent_text1, parent_text2, original_text
    ):
        """Check if `new_text` that has been produced by performing crossover
        between `parent_text1` and `parent_text2` aligns with the constraints.

        Args:
            new_text (AttackedText): Text produced by crossover operation
            parent_text1 (AttackedText): Parent text of `new_text`
            parent_text2 (AttackedText): Second parent text of `new_text`
            original_text (AttackedText): Original text
        Returns:
            `True` if `new_text` meets the constraints. If otherwise, return `False`.
        """
        if "last_transformation" in new_text.attack_attrs:
            previous_text = (
                parent_text1
                if "last_transformation" in parent_text1.attack_attrs
                else parent_text2
            )
            passed_constraints = self._check_constraints(
                new_text, previous_text, original_text=original_text
            )
            return passed_constraints
        else:
            # `new_text` has not been actually transformed, so return True
            return True


    def _initialize_population(self, initial_result, pop_size):
        words = initial_result.attacked_text.words
        num_candidate_transformations = np.zeros(len(words))
        transformed_texts = self.get_transformations(
            initial_result.attacked_text, original_text=initial_result.attacked_text
        )
        for transformed_text in transformed_texts:
            diff_idx = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
            num_candidate_transformations[diff_idx] += 1

        # Just b/c there are no replacements now doesn't mean we never want to select the word for perturbation
        # Therefore, we give small non-zero probability for words with no replacements
        # Epsilon is some small number to approximately assign small probability
        min_num_candidates = np.amin(num_candidate_transformations)
        epsilon = max(1, int(min_num_candidates * 0.1))
        for i in range(len(num_candidate_transformations)):
            num_candidate_transformations[i] = max(
                num_candidate_transformations[i], epsilon
            )

        population = []
        for _ in range(pop_size):
            pop_member = PopulationMember(
                initial_result.attacked_text,
                initial_result,
                attributes={
                    "num_candidate_transformations": np.copy(
                        num_candidate_transformations
                    )
                },
            )
            # Mutate `pop_member` 
            pop_member = self.mutation(pop_member, initial_result)
            population.append(pop_member)

        return population

    def modify_population_member(self, pop_member, new_text, new_result, word_idx):
        """Modify `pop_member` by returning a new copy with `new_text`,
        `new_result`, and `num_candidate_transformations` altered appropriately
        for given `word_idx`"""
        num_candidate_transformations = np.copy(
            pop_member.attributes["num_candidate_transformations"]
        )
        num_candidate_transformations[word_idx] = 0
        return PopulationMember(
            new_text,
            result=new_result,
            attributes={"num_candidate_transformations": num_candidate_transformations},
        )
    
    def get_word_select_prob_weights(self, pop_member):
        """Get the attribute of `pop_member` that is used for determining
        probability of each word being selected for matation."""
        return pop_member.attributes["num_candidate_transformations"]
        
    def check_transformation_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return [
            "pop_size",
            "iters",
            "give_up_if_no_improvement",
            "post_crossover_check",
            "max_crossover_retries",
        ]
