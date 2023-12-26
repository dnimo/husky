from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from tools import valuations


class valuationsTest(absltest.TestCase):

    def testBLEU(self):
        candidates = [['It is a guide to action that ensures that the military always obeys the commands of the party']]

        references = [['It is a guide to action that ensures that the military will forever heed party commands'],
                      ['it is the guiding principe which guarantees the military always being under the command of the party'],
                      ['it is the practical guide for the army always to heed the directions of the party']
                      ]

        candidates = [[s.split() for s in candidate] for candidate in candidates]
        references = [[s.split() for s in reference] for reference in references]
        mybleu = valuations.Bleu(2).valuate(references, candidates)
        print(mybleu)

    def testRouge(self):
        myRouge = valuations.Rouge(['rouge1'])
        score = myRouge.valuate("this is a  test", "this is a test")
        print(score)


if __name__ == '__main__':
    absltest.main()