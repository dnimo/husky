import abc
from . import bleu
from . import rouge_scorer


class Valuations(abc.ABC):
    @abc.abstractmethod
    def valuate(self, target, prediction):
        raise NotImplementedError("valuations must override valuate method")


class Bleu(Valuations):
    def __init__(self, n_gram=1):
        self._n_gram = n_gram

    def valuate(self, target, prediction):
        return bleu.BLEU(self._n_gram).evaluate(prediction, target)


class Rouge(Valuations):

    def valuate(self, target, prediction):
        return rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL']).score(
            target=target,
            prediction=prediction
        )
