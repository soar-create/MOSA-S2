"""
Greedy Word Swap with Word Importance Ranking
===================================================


When WIR method is set to ``unk``, this is a reimplementation of the search
method from the paper: Is BERT Really Robust?

A Strong Baseline for Natural Language Attack on Text Classification and
Entailment by Jin et. al, 2019. See https://arxiv.org/abs/1907.11932 and
https://github.com/jind11/TextFooler.
"""

import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)

import math
import random
from textattack.metrics import Metric
#from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
#use = UniversalSentenceEncoder()


class GreedyWordSwapWIR(SearchMethod):
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
            #print(leave_one_texts)
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])
            #print(index_scores)

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
            #index_order = (index_scores).argsort()
        return index_order, search_over
        #indices_to_modify=[]
    """
    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        #print(attacked_text)
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        print(index_order)
        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i:i+4]],
            )
            #print(transformed_text_candidates)
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            print("results[0].score")
            print(results[0].score)
            #print(type(results))  <class list>
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
                #index_order, search_over = self._get_index_order(cur_result.attacked_text)
                #print("cur_result.score")
                #print(cur_result.score)
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                        print("best_result.score")
                return best_result

        return cur_result
    """
    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text
        
        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        #print(index_order)
        # Starts Simulated Annealing
        inputlen=len(index_order)
        T =1000 # initial temperature
        Tmin = (math.pow(0.9,inputlen+3))*T  # lowest temperature
        #k =round(inputlen*0.6)  # internal iterations
        #k=12
    
        if inputlen < 21:
            k=14
        else:
            k=15
        
        #print(k)
        x = initial_result  # initialize x as original input text
        xin = initial_result  # initialize xin as original input text in cycle
        t = 0  # time,external iterations
        
        while T >= Tmin:
            index_order, search_over = self._get_index_order(x.attacked_text)
            weights=[]
            iw=0
            
            if t==0:
                weights=torch.Tensor([0.7,0.3])#第一次删除在最不重要与次不重要里面选
                weightsam=torch.multinomial(weights,1)
                listk=weightsam.numpy()
    
            elif t==inputlen-2:
                listk=[0]
            
            else:
                iw=0
                while iw  < len(index_order):
                    
                    if 1-(1/len(index_order))*iw<0 or iw>min(2*t,len(index_order)):
                        w=0
                    else:
                        w=1-(1/len(index_order))*iw
                    
                    #iw=iw*0.95
                    weights.append(w)
                    iw+=1
                
                weights=torch.Tensor(weights)
                #weightsam=torch.multinomial(weights,min(k,len(weights)))
                weightsam=torch.multinomial(weights,k,replacement=True)
                listk=weightsam.numpy()
            #listk=random.sample(range(0,min(t+1,k,len(index_order))),min(k,len(index_order),t+1))#第一次删除取0或1不重要的
            #print(weights)
            #listk=random.sample(range(0,3*t),min(k,3*t+1)
            #print(index_order)
            for indice in listk:
                y=xin.score
                if indice >= len(index_order):
                    #print("continue due to indice out of range")
                    continue
                moves = self.get_transformations(
                            x.attacked_text,
                            original_text=initial_result.attacked_text,
                            indices_to_modify=[index_order[indice]]
                            )
                # Skip words without candidates
                if len(moves) == 0:
                    continue
                xNew, _ = self.get_goal_results(moves)
                xNew_sorted = sorted(xNew, key=lambda x: -x.score)
                #yNew = self._aim_function(initial_result, xNew_sorted[0])
                yNew=xNew_sorted[0].score
                #print(xNew_sorted[0].score)
                #print("差为：")
                #print(yNew-y)
                d=yNew-y
                #print(d)
                if xNew_sorted[0].score!=0:
                    if yNew-y > 0:
                        xin = xNew_sorted[0]
                    else:
                    # metropolis principle
                        #print(d)
                        """
                        if (-d)>0.01 and (-d)<=0.1:
                            d=d/100
                        elif (-d)>0.001 and (-d)<=0.01:
                            d=d/10
                        
                        if t<=0.3*inputlen:
                            p = math.exp(-((d)*(-1000000))/T)
                        elif t>0.3*inputlen and t<0.8*inputlen:
                            p = math.exp(-((d)*(-900000))/T)
                        else:
                            p = math.exp(-((d)*(-1100000))/T)
                       
                        if t<=5:
                            p = math.exp(-((d)*(-900000))/T)
                        elif t>5 and t<12:
                            p = math.exp(-((d)*(-1000000))/T)
                        else:
                            p = math.exp(-((d)*(-450000))/T)
                       
                        if len(initial_result.attacked_text.words)>16:
                            p=math.exp(-(60/T))
                        elif len(initial_result.attacked_text.words)<17 and len(initial_result.attacked_text.words)>9:
                            p = math.exp(-(100/T))
                        else:
                            p = math.exp(-(150/T))
                        
                        if inputlen > 20:
                            p = math.exp(-((d)*(-1000000))/T)
                        if inputlen <=20 and inputlen >=10:
                            p = math.exp(-((d)*(-990000))/T) 
                        else:
                            p = math.exp(-((d)*(-950000))/T)
                        """
                        p = math.exp(-((-d)*1000000)/T)
                        #print("T为：")
                        #print(T)
                        #print("p为：")
                        #print(p)
                        r = np.random.uniform(low=0, high=1)
                        #r=torch.rand(1).item()
                        #print(r)
                        if r < p:
                            xin = xNew_sorted[0]
                            #print("跳出")
                else:
                    continue
            t += 1
            T = 0.9*T
            #分段体现在降温函数上也可，跳出概率函数中的T改为T1即可
            #T = 3/(5*t)
            x=xin
            #semantics_score=use._sim_score(x.attacked_text)
        

            #print(x.attacked_text.tests)
        
        return x
        #f返回x1、x2，比较取得分最高
    """
    def perform_search(self, initial_result):
        beam_width=5
        index_order, search_over = self._get_index_order(initial_result.attacked_text)
        beam = [initial_result.attacked_text]
        best_result = initial_result
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                index_order, search_over = self._get_index_order(text)
                transformations = self.get_transformations(
                    text, 
                    original_text=initial_result.attacked_text,
                    indices_to_modify=[index_order[:beam_width]]
                )
                potential_next_beam += transformations

            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            if results[scores.argmax()].score!=0:
                best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: beam_width]
            beam = [potential_next_beam[i] for i in best_indices]
            #print(beam)
        return best_result
    """ 
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
