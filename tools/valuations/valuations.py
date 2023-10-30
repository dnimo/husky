import abc
import bleu
import rouge_scorer


class valuations(abc.ABC):
    @abc.abstractmethod
    def valuate(self, target, prediction):
        raise NotImplementedError("valuations must override valuate method")


class Bleu(valuations):
    def __init__(self, n_gram=1):
        self._n_gram = n_gram

    def valuate(self, target, prediction):
        return bleu.BLEU(self._n_gram).evaluate(prediction, target)


class Rouge(valuations):
    def __init__(self, rouge_type, use_stemmer=False, split_summarise=False, tokenizer=None):
        self.tokenizer = tokenizer
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self.split_summarise = split_summarise

    def valuate(self, target, prediction):
        return rouge_scorer.RougeScorer(rouge_types=self.rouge_type, use_stemmer=self.use_stemmer,
                                        split_summaries=self.split_summarise, tokenizer=self.tokenizer).score(
            target=target,
            prediction=prediction

        )
