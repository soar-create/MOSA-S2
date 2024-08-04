"""
Managing Attack Logs.
========================
"""

import numpy as np

from textattack.attack_results import FailedAttackResult, SkippedAttackResult

from . import CSVLogger, FileLogger, VisdomLogger, WeightsAndBiasesLogger
from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.metrics import Metric
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
use = UniversalSentenceEncoder()
import matplotlib.pyplot as plt

class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    def __init__(self):
        self.loggers = []
        self.results = []

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self):
        self.loggers.append(WeightsAndBiasesLogger())

    def disable_color(self):
        self.loggers.append(FileLogger(stdout=True, color_method="file"))

    def add_output_file(self, filename, color_method):
        self.loggers.append(FileLogger(filename=filename, color_method=color_method))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def log_result(self, result):
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result)

    def log_results(self, results):
        """Logs an iterable of ``AttackResult`` objects on each of
        `self.loggers`."""
        for result in results:
            self.log_result(result)
        self.log_summary()

    def log_summary_rows(self, rows, title, window_id):
        for logger in self.loggers:
            logger.log_summary_rows(rows, title, window_id)

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack_name, model_name):
        # @TODO log a more complete set of attack details
        attack_detail_rows = [
            ["Attack algorithm:", attack_name],
            ["Model:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")
    
    def log_summary(self):
        total_attacks = len(self.results)
        if total_attacks == 0:
            return
        # Count things about attacks.
        confidence_score= np.zeros(len(self.results))
        confidence_score_output= np.zeros(len(self.results))
        all_num_words = np.zeros(len(self.results))
        all_num_premisewords = np.zeros(len(self.results))
        all_num_words_output = np.zeros(len(self.results))
        all_num_nonewords_output = np.zeros(len(self.results))
        all_num_none_output = np.zeros(len(self.results))
        all_num_premisewords_output = np.zeros(len(self.results))
        all_semantic_score_eq = np.zeros(len(self.results))
        all_semantic_score_noeq = np.zeros(len(self.results))
        ori_semantic_score_eq = np.zeros(len(self.results))
        ori_semantic_score_eq = np.zeros(len(self.results))
        all_semantic_score=np.zeros(len(self.results))
        perturbed_word_percentages = np.zeros(len(self.results))
        perturbed_posnoneword_percentages=np.zeros(len(self.results))
        num_words_changed_until_success = np.zeros(
            2 ** 16
        )  # @ TODO: be smarter about this
        failed_attacks = 0
        skipped_attacks = 0
        successful_attacks = 0
        max_words_changed = 0
        for i, result in enumerate(self.results):
            all_num_words[i] = len(result.original_result.attacked_text.words)
 
            if isinstance(result, FailedAttackResult):
                continue
            elif isinstance(result, SkippedAttackResult):
                continue
            else:
                all_num_words_output[i] = len(result.perturbed_result.attacked_text.words)
                if len(result.perturbed_result.attacked_text.words)!=2:
                    all_num_nonewords_output[i]=len(result.perturbed_result.attacked_text.words)
                    all_num_none_output[i]=1
                    
                confidence_score[i]=result.original_result.raw_output.max()
                confidence_score_output[i]=result.perturbed_result.raw_output.max()
                if result.perturbed_result!=result.original_result:
                    all_semantic_score[i]=use._sim_score(result.original_result.attacked_text,result.perturbed_result.attacked_text)

            if isinstance(result, FailedAttackResult):
                failed_attacks += 1
                continue
            elif isinstance(result, SkippedAttackResult):
                skipped_attacks += 1
                continue
            else:
                successful_attacks += 1
            num_words_changed = result.original_result.attacked_text.words_diff_num(
                result.perturbed_result.attacked_text
            )
            num_words_changed_until_success[num_words_changed - 1] += 1
            max_words_changed = max(
                max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0
            perturbed_word_percentages[i] = perturbed_word_percentage 
            
            if not isinstance(result, (SkippedAttackResult,FailedAttackResult)):
                word_pos_originaltext=0
                num_words_changed = result.original_result.attacked_text.words_diff_num(result.perturbed_result.attacked_text)
                
                for w in range(len(result.original_result.attacked_text.words)):
                #for w in range(len(result.original_result.attacked_text.words)-len(result.original_result.attacked_text.premisewords)):#only modify the premise
                    word_pos = result.original_result.attacked_text.pos_of_word_index(w)
                    #word_pos = result.original_result.attacked_text.pos_of_word_index(w+len(result.original_result.attacked_text.premisewords))
                    word_rep=result.original_result.attacked_text.words[w]
                    #word_rep=result.original_result.attacked_text.words[w+len(result.original_result.attacked_text.premisewords)]
                    pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}
                    word_pos = pos_dict.get(word_pos, None)
                    if word_pos!=None:
                        word_pos_originaltext+=1
                if word_pos_originaltext!=0:
                    perturbed_posnoneword_percentage=(num_words_changed*100.0)/word_pos_originaltext
                    perturbed_posnoneword_percentages[i]=perturbed_posnoneword_percentage

        # Original classifier success rate on these samples.
        original_accuracy = (total_attacks - skipped_attacks) * 100.0 / (total_attacks)
        original_accuracy = str(round(original_accuracy, 2)) + "%"

        # New classifier success rate on these samples.
        accuracy_under_attack = (failed_attacks) * 100.0 / (total_attacks)
        accuracy_under_attack = str(round(accuracy_under_attack, 2)) + "%"

        # Attack success rate.
        if successful_attacks + failed_attacks == 0:
            attack_success_rate = 0
        else:
            attack_success_rate = (
                successful_attacks * 100.0 / (successful_attacks + failed_attacks)
            )
        attack_success_rate = str(round(attack_success_rate, 2)) + "%"

        perturbed_word_percentages = perturbed_word_percentages[
            perturbed_word_percentages > 0
        ]
        #print(perturbed_word_percentages)
        average_perc_words_perturbed = perturbed_word_percentages.mean()
        average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"
            
        average_perc_posnone_words_perturbed = sum(perturbed_posnoneword_percentages)/len(np.nonzero(perturbed_posnoneword_percentages)[0])
        average_perc_posnone_words_perturbed = str(round(average_perc_posnone_words_perturbed, 2)) + "%"

        average_num_words = all_num_words.mean()
        average_num_words = str(round(average_num_words, 5))
        
        
        average_num_words_output = sum(all_num_words_output)/len(np.nonzero(all_num_words_output)[0])
        average_num_words_output = str(round(average_num_words_output, 5))
        
        #print(all_semantic_score)
        average_semantic_score = sum(all_semantic_score)/len(np.nonzero(all_semantic_score)[0])
        average_semantic_score = str(round(average_semantic_score, 5))

        average_confidence_score = sum(confidence_score)/len(np.nonzero(confidence_score)[0])
        average_confidence_score = str(round(average_confidence_score, 4))
        
        
        average_confidence_score_output = sum(confidence_score_output)/len(np.nonzero(confidence_score_output)[0])
        average_confidence_score_output = str(round(average_confidence_score_output, 4))
        #use_stats = USEMetric().calculate(self.results)
        summary_table_rows = [
            ["Number of successful attacks:", str(successful_attacks)],
            ["Number of failed attacks:", str(failed_attacks)],
            ["Number of skipped attacks:", str(skipped_attacks)],
            ["Original accuracy:", original_accuracy],
            ["Accuracy under attack:", accuracy_under_attack],
            ["Attack success rate:", attack_success_rate],
            ["Average perturbed word %:", average_perc_words_perturbed],
            ["Average num. words per input:", average_num_words],
            ["Average num. words per output:", average_num_words_output],
            ["Average confidence_score_input:", average_confidence_score],
            ["Average confidence_score_output:", average_confidence_score_output],
            ["Average perturbed psonone word %:", average_perc_posnone_words_perturbed],
            ["Average Attack USE Score:", average_semantic_score],
        ]

        num_queries = np.array(
            [
                r.num_queries
                for r in self.results
                if not isinstance(r, SkippedAttackResult)
            ]
        )
        avg_num_queries = num_queries.mean()
        avg_num_queries = str(round(avg_num_queries, 2))
        summary_table_rows.append(["Avg num queries:", avg_num_queries])
        self.log_summary_rows(
            summary_table_rows, "Attack Results", "attack_results_summary"
        )
        # Show histogram of words changed.
        numbins = max(max_words_changed, 10)
        for logger in self.loggers:
            logger.log_hist(
                num_words_changed_until_success[:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )
