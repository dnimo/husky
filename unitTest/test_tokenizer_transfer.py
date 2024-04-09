import unittest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pd2_model
from transformers import GPTNeoXTokenizerFast

class MyTestCase(unittest.TestCase):
    def test_something(self):
        kuhp_sp_tokenizer = spm.SentencePieceProcessor(model_file=r'C:\Users\KuoChing\workspace\husky\data\tokenizer\bert_bpe_vocab_40000.model')
        kuhp_spm = sp_pd2_model.ModelProto()
        kuhp_spm.ParseFromString(kuhp_sp_tokenizer.serialized_model_proto())

        kuhp_spm_tokens_set = set(p.piece for p in kuhp_spm.pieces)

        print("length of kuhp_spm_tokens_set: ", len(kuhp_spm_tokens_set))

        #load calm tokenizer

        calmTokenizer = GPTNeoXTokenizerFast.from_pretrained("cyberagent/open-calm-7b")
        calmTokenizer_spm = sp_pd2_model.ModelProto()
        calmTokenizer_spm.ParseFromString(calmTokenizer.sp_model.serialized_model_proto())

        calmTokenizer_spm_tokens_set = set(p.piece for p in calmTokenizer_spm.pieces)

        print("length of calmTokenizer_spm_tokens_set: ", len(calmTokenizer_spm_tokens_set))

        already_in_calm = set()
        not_in_calm = set()
        for token in kuhp_spm.pieces:
            if token.piece not in calmTokenizer_spm_tokens_set:
                new_p = sp_pd2_model.ModelProto.SentencePiece()
                new_p.piece = token.piece
                new_p.score = 0
                calmTokenizer_spm.pieces.append(new_p)
                calmTokenizer_spm_tokens_set.add(token.piece)
                not_in_calm.add(token.piece)
            else:
                already_in_calm.add(token.piece)
        print("already_in_calm: ", len(already_in_calm))
        print("not_in_calm: ", len(not_in_calm))
        print("New length of calmTokenizer_spm_tokens_set: ", len(calmTokenizer_spm.pieces))

        with open("Newtokenizer/Newtokenizer.model", "wb") as f:
            f.write(calmTokenizer_spm.SerializeToString())

    newTokenizer = GPTNeoXTokenizerFast("Newtokenizer/Newtokenizer.model")
    newTokenizer.save_pretrained("Newtokenizer")



        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
