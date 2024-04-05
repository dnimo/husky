"""Tests for bleu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tools.valuations import bleu


class BLEUTest(absltest.TestCase):
    def testBLEUEvaluate(self):
        bleu_ = bleu.BLEU(4)

        candidate = 'It is a guide to action that ensures that the military always obeys the commands of the party'

        reference = 'It is a guide to action that ensures that the military will forever heed party commands'

        print(candidate)
        print(len(reference))

        score = bleu_.evaluate(candidate, reference)
        print(score)
        self.assertAlmostEquals(0.597497, score, places=3)


if __name__ == '__main__':
    absltest.main()
