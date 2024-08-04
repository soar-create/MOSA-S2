"""
Multi-Objective Simulated Annealing with Word Importance Ranking
===================================================

"""

import numpy as np
import torch
import random
from torch.nn.functional import softmax
from itertools import permutations

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)

import math
import random
#from textattack.metrics import Metric
#from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
#use = UniversalSentenceEncoder()


class MOSA(SearchMethod):
    """An attack that greedily chooses from a list of possible perturbations in
    order of index, after ranking indices by importance.

    Args:
        wir_method: method for ranking most important words
        model_wrapper: model wrapper used for gradient-based ranking
    """

    def __init__(self, wir_method="unk"):
        self.wir_method = wir_method

    def _get_index_order(self, initial_text):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""
        len_text = len(initial_text.words)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, "[UNK]") for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in range(len_text):
                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in range(len_text)
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score[1] for result in leave_one_results])
            
        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(initial_text.num_words)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, word in enumerate(initial_text.words):
                matched_tokens = word2token_mapping[i]
                #print(matched_tokens)
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    if matched_tokens!=None:
                    	matched_tokens=None
                    else:
                    	agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    	index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = np.arange(len_text)
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = (index_scores).argsort()[::-1]
        return index_order, search_over
    
    def fastNonDominatedSort(self, pop):
        fronts = [[]]
        for p in pop:
            p.dominatedSolutions = []
            p.dominationCount = 0
            for q in pop:
                if q.score[0]< p.score[0] and q.score[1] < p.score[1]:
                    p.dominatedSolutions.append(q)
                elif p.score[0] < q.score[0] and p.score[1] < q.score[1]:
                    p.dominationCount += 1
            if p.dominationCount == 0:
                p.rank = 0
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            nextFront = []
            for p in fronts[i]:
                for q in p.dominatedSolutions:
                    q.dominationCount -= 1
                    if q.dominationCount == 0:
                        q.rank = i + 1
                        nextFront.append(q)
            i += 1
            fronts.append(nextFront)
        return fronts[:-1]

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}
        # Starts Simulated Annealing
        inputlen=len(attacked_text.words)
        T = 1000 # initial temperature
        Tmin = (math.pow(0.9,inputlen+3))*T  # lowest temperature
        #k =round(inputlen*0.6)  # internal iterations
        #k=min(16,9+int(inputlen/5))
        #k=14
        #k=12
        if inputlen < 25:
            k=14
        else:
            k=15
        adversarial_examples=[]
        x = initial_result  # initialize x as original input text
        xin = initial_result  # initialize xin as original input text in cycle
        t = 0  # time,external iterations
        while T >= Tmin:
            index_order, search_over = self._get_index_order(x.attacked_text)
            for i in index_order:
                word_to_replace = x.attacked_text.words[i]
                if word_to_replace==attacked_text.words[i]:
                    word_to_replace_pos = attacked_text.pos_of_word_index(i)
                else:
                    word_to_replace_pos = x.attacked_text.pos_of_word_index(i)
                word_pos = pos_dict.get(word_to_replace_pos, None)
                if word_pos==None:
                    index_order=index_order.tolist()
                    index_order.remove(i)
                    index_order=np.array(index_order)
            if len(index_order)==0:
                T = 0.9*T
                continue
            weights=[]
            move1=[]
            move2=[]
            iw=0       
            if t==0:
                xin.rank=100
                weights=torch.Tensor([1,0])
                weightsam=torch.multinomial(weights,1)
                listk=weightsam.numpy()
                for indice in listk:
                    moves = self.get_transformations(
                                x.attacked_text,
                                original_text=initial_result.attacked_text,
                                indices_to_modify=[index_order[indice]]
                                )
                    if len(moves) == 0:
                        continue
                    xNew, _ = self.get_goal_results(moves)
                    xNew_sorted = sorted(xNew, key=lambda x: -x.score[1])
                    if xNew_sorted[0].score!=[]:
                        x=xNew_sorted[0]
            else:
                iw=0
                while iw  < len(index_order):
                    
                    if (1-(1/len(index_order))*iw)<0 or iw>min(2*t,len(index_order)):
                        w=0
                    else:
                        w=1-(1/len(index_order))*iw
                    weights.append(w)
                    iw+=1
                weights=torch.Tensor(weights)
                weightsam=torch.multinomial(weights,min(k,2*t),replacement=True)
                listk=weightsam.numpy()
                for indice in range(len(listk)):
                    if indice >= len(index_order):
                        continue
                    moves = self.get_transformations(
                                x.attacked_text,
                                original_text=initial_result.attacked_text,
                                indices_to_modify=[index_order[listk[indice]]]
                                )
                    # Skip words without candidates
                    if len(moves) == 0:
                        continue
                    xNew, _ = self.get_goal_results(moves)
                    xNew_sorted = sorted(xNew, key=lambda x: -x.score[1])
                    move1.append(xNew_sorted[0])
                    for indice2 in listk[indice+1:]:
                        moves = self.get_transformations(
                                    xNew_sorted[0].attacked_text,
                                    original_text=initial_result.attacked_text,
                                    indices_to_modify=[index_order[indice2]]
                                    )
                        if len(moves)==0:
                            continue
                        xNew, _ = self.get_goal_results(moves)
                        xNew_sorted = sorted(xNew, key=lambda x: -x.score[1])
                        move2.append(xNew_sorted[0]) 
                    move_c=move1+move2 
                    fronts = self.fastNonDominatedSort(move_c) 
                    random.shuffle(move_c)
                    if len(move_c)==0:
                        continue
                    for x_c in move_c:
                        if x_c.score!=[0.0, 0.0]:
                            if x_c.rank < xin.rank:
                                xin=x_c
                            else:
                                p = math.exp(-((x_c.rank-xin.rank)*400)/T)
                                r = np.random.uniform(low=0, high=1)
                                if r < p:
                                    xin=x_c           
                        else:
                            continue
            t += 1
            T = 0.9*T
            x=xin
        
        return x
        

    def check_transformation_compatibility(self, transformation):
        """Since it ranks words by their importance, GreedyWordSwapWIR is
        limited to word swap and deletion transformations."""
        return transformation_consists_of_word_swaps_and_deletions(transformation)

    @property
    def is_black_box(self):
        if self.wir_method == "gradient":
            return False
        else:
            return True

    def extra_repr_keys(self):
        return ["wir_method"]
