##########################################################################################
# LMBNC: Language Model Based N-Gram Corrector
#
# Copyright 2020 by Targoman Co. Pjc.
#                                                                         
# LMBNC is free software: you can redistribute it and/or modify         
# it under the terms of the GNU Lesser General Public License as published 
# by the Free Software Foundation, either version 3 of the License, or     
# (at your option) any later version.                                      
#                                                                         
# LMBNC is distributed in the hope that it will be useful,              
# but WITHOUT ANY WARRANTY; without even the implied warranty of           
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            
# GNU Lesser General Public License for more details.                      
# You should have received a copy of the GNU Lesser General Public License 
# along with Targoman. If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################################
# @author Zakieh Shakeri <z.shakeri@targoman.com>
# @author Behrooz Vedadian <vedadian@targoman.com>
##########################################################################################

import re

import numpy as np
import torch
from fairseq.models.transformer_lm import TransformerLanguageModel

class LMBNC:

    def __init__(self, lm_model_path):
        self.lm =  TransformerLanguageModel.from_pretrained(lm_model_path, 'checkpoint_best.pt', tokenizer='moses')
        self.lm.eval()
        if torch.cuda.is_available():
            self.lm.cuda()
    
    def __get_ngram_dictionaries(self):
        unigrams = {}
        bigrams = {}
        for tokens in self.corpus:
            for token in tokens:
                if token not in unigrams:
                    unigrams[token] = 1
                else:
                    unigrams[token] += 1
            for t0, t1 in zip(tokens[:-1], tokens[1:]):
                bigram = '{0} {1}'.format(t0, t1)
                if bigram not in bigrams:
                    bigrams[bigram] = 1
                else:
                    bigrams[bigram] += 1
        return unigrams, bigrams

    def load_corpus(self, corpus_file_path):
        self.corpus = [l.strip().split() for l in open(corpus_file_path, encoding="utf8")]
        self.unigrams, self.bigrams = self.__get_ngram_dictionaries()

    @staticmethod
    def __contains_digit(word):
        p = re.compile(r'(.*)[(\d)+](.*)', re.UNICODE)
        result = p.match(word)
        return result
    
    def __get_unigram_parts(self, word):
        parts = []
        word_occurrence_count = int(
            self.unigrams[word]) if word in self.unigrams else 0
        try:
            if len(word) > 1 and not LMBNC.__contains_digit(word):
                i = len(word) - 2
                part1 = word[:i + 1]
                part2 = word[i + 1:]
                while (i >= 0):
                    try:
                        if int(self.unigrams[part1]) > 2 and int(self.unigrams[part2]) > 2:
                            if int(self.bigrams[part1 + ' ' + part2]) > word_occurrence_count:
                                parts.append(part1 + ' ' + part2)
                    except KeyError:
                        pass
                    i -= 1
                    part1 = word[:i + 1]
                    part2 = word[i + 1:]
        except:
            pass
        parts.append(word)
        return parts
    
    def __extract_alternatives(self, tokens):
        if len(tokens) == 0:
            return ['']
        first_word_parts = self.__get_unigram_parts(tokens[0])
        if len(tokens) == 1:
            return first_word_parts
        else:
            alternatives = [part + ' ' + alternative
                            for alternative in self.__extract_alternatives(tokens[1:])
                            for part in first_word_parts]
            if (tokens[0] + tokens[1] in self.unigrams) and \
                    int(self.unigrams[tokens[0] + tokens[1]]) > 2000 and \
                    int(self.unigrams[tokens[0] + tokens[1]]) > int(self.bigrams[tokens[0] + ' ' + tokens[1]]):
                alternatives.extend([tokens[0] + tokens[1] + ' ' + alternative
                                     for alternative in self.__extract_alternatives(tokens[2:])])
            return alternatives

    def __calc_score(self, alternatives):
        scores = np.array([
            item['positional_scores'].mean().neg().exp()
            for item in self.lm.score(alternatives)
        ])
        return scores

    def correct_ngrams(self):
        for index, tokens in enumerate(self.corpus):
            alternatives = self.__extract_alternatives(tokens)
            scores = self.__calc_score(alternatives)
            if len(scores) > 0:
                best_score_index = scores.argmin()
                self.corpus[index] = alternatives[best_score_index].split()
            else:
                # Keep the original sentence form
                pass

    def save_corpus(self, corpus_file_path):
        with open(corpus_file_path, 'wt', encoding="utf8") as output_corpus:
            for tokens in self.corpus:
                output_corpus.write(' '.join(tokens) + '\n')
