"""Tests for bleu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import bleu


class BLEUTest(absltest.TestCase):
    def testBLEUEvaluate(self):
        bleu_ = bleu.BLEU(2)

        candidates = [['It is a guide to action that ensures that the military always obeys the commands of the party']]

        references = [['It is a guide to acttion that ensures that the military will forever heed party commands'], 
                      ['it is the guiding principe which guarantees the military always being under the command of the party'], 
                      ['it is the practical guide for the army always to heed the directions of the party']
                      ]

        candidates = [[s.split() for s in candidate] for candidate in candidates]
        references = [[s.split() for s in reference] for reference in references]

        score = bleu_.evaluate(candidates, references)
        print(score[0])
        self.assertAlmostEquals(0.56011, score[0], places=5)


if __name__ == '__main__':
  absltest.main()