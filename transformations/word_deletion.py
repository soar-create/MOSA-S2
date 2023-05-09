from .transformation import Transformation
import string

class WordDeletion(Transformation):
    """An abstract class that takes a sentence and transforms it by deleting a
    single word.

    letters_to_insert (string): letters allowed for insertion into words
    """

    def _get_transformations(self, current_text, indices_to_modify):
        # words = current_text.words
        transformed_texts = []
        #len_premise=0
        #len_premise=len(current_text.premisewords)
        #for c in current_text.premisewords:
            #if c.isspace():
            #if not c.isalpha():
             #   len_premise=len_premise+1
        #print(len_premise)
        if len(current_text.words) > 1: 
            #for i in np.nditer(indices_to_modify):
            for i in indices_to_modify: 
                
                if len(current_text.premisewords)==1 and i==0:
                    continue
                if len(current_text.words)==len(current_text.premisewords)+1 and i >= len(current_text.premisewords):
                    continue
                else:
                
                    transformed_texts.append(current_text.delete_word_at_index(i))
                #transformed_texts.append(current_text.delete_word_at_index(i+1))
                #print(type(indices_to_modify))
                #print(type(current_text))<class 'textattack.shared.attacked_text.AttackedText'>
        return transformed_texts
