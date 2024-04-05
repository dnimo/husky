from . import myMeCab


def is_string(s):
    return isinstance(s, str)


class BasicTokenizer(object):
    """basic Tokenizer
    """

    def __init__(self, token_start='[CLS]', token_end='[SEP]'):
        """
        Init BasicTokenizer
        """
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = token_start
        self._token_end = token_end

    def tokenize(self, text, maxlen=512):
        """
        tokenize function
        output -> [tokens, tokens]
        """
        tokens = self._tokenize(text)
        if self._token_start is not None:
            tokens.insert(0, self._token_start)
        if self._token_end is not None:
            tokens.append(self._token_end)

        if maxlen is not None:
            index = int(self._token_end is not None) + 1
            self.truncate_sequence(maxlen, tokens, None, -index)

        return tokens

    def token_to_id(self, token):
        """
        one token to id
        """
        raise NotImplementedError

    def tokens_to_ids(self, tokens):
        """
        tokens to ids
        input -> [tokens, tokens]
        output -> [ids, ids]
        """
        return [self.token_to_id(token) for token in tokens]

    def truncate_sequence(
            self, maxlen, first_sequence, second_sequence=None, pop_index=-1
    ):
        """
        Truncate the sequence to maxlen.
        """
        if second_sequence is None:
            second_sequence = []

        while True:
            total_length = len(first_sequence) + len(second_sequence)
            if total_length <= maxlen:
                break
            elif len(first_sequence) > len(second_sequence):
                first_sequence.pop(pop_index)
            else:
                second_sequence.pop(pop_index)

    def encode(
            self, first_text, second_text=None, maxlen=None, pattern='S*E*E'
    ):
        """
        output token id and segment id
        """
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text

        if second_text is None:
            second_tokens = None
        elif is_string(second_text):
            if pattern == 'S*E*E':
                idx = int(bool(self._token_start))
                second_tokens = self.tokenize(second_text)[idx:]
            elif pattern == 'S*ES*E':
                second_tokens = self.tokenize(second_text)
        else:
            second_tokens = second_text

        if maxlen is not None:
            self.truncate_sequence(maxlen, first_tokens, second_tokens, -2)

        first_token_ids = self.tokens_to_ids(first_tokens)
        first_segment_ids = [0] * len(first_token_ids)

        if second_text is not None:
            second_token_ids = self.tokens_to_ids(second_tokens)
            second_segment_ids = [1] * len(second_token_ids)
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def id_to_token(self, i):
        """id to token
        """
        raise NotImplementedError

    def ids_to_tokens(self, ids):
        """id sequence to token sequence
        """
        return [self.id_to_token(i) for i in ids]

    def decode(self, ids):
        """decode id sequence
        """
        raise NotImplementedError


class Mecab(BasicTokenizer):
    def tokenize(self, text, stem=False, maxlen: int = 512):
        words = myMeCab.tokenize(text=text, stemmer=stem)

        return words

class spTokenizer(BasicTokenizer):
    # TODO: implement spTokenizer

    def __init__(self, sp_model):
        self.sp_model = sp_model
        self._token_pad = '[PAD]'
        self._token_unk = '[UNK]'
        self._token_mask = '[MASK]'
        self._token_start = '[CLS]'
        self._token_end = '[SEP]'

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def id_to_token(self, i):
        return self.sp_model.IdToPiece(i)

    def decode(self, ids):
        return self.sp_model.DecodeIds(ids)