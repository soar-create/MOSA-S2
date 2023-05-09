"""

Determine successful in rubbish examples generation
----------------------------------------------------
"""


from .classification_goal_function import ClassificationGoalFunction


class Rubbishexample(ClassificationGoalFunction):
    """Rubbish examples generation on classification models which attempts to maximize
    the confidence of the original label and word modifcation rate.
    """

    def __init__(self, *args, target_max_score=None, **kwargs):
        self.target_max_score = target_max_score
        super().__init__(*args, **kwargs)

    def _is_goal_complete(self, model_output, _):
        if self.target_max_score:
            return model_output[self.ground_truth_output] < self.target_max_score
        elif (model_output.numel() == 1) and isinstance(
            self.ground_truth_output, float
        ):
            return abs(self.ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != self.ground_truth_output

    def _get_score(self, model_output, attacked_text):
        cur_words = attacked_text.words
        initial_words = self.initial_attacked_text.words
        initial_num_words=self.initial_attacked_text.num_words
        if self.ground_truth_output != model_output.argmax():
            return 0
        if (model_output.numel() == 1) and isinstance(self.ground_truth_output, float):
            return abs(model_output.item() - self.ground_truth_output)
        else:
            num_words_score = len(set(cur_words)-set(initial_words))
            model_score = model_output[self.ground_truth_output]
            SCORE=((num_words_score + 2.5*model_score) / initial_num_words)*10000
            return SCORE
            
