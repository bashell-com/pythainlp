# -*- coding: utf-8 -*-
from pythainlp.corpus import wordnet

def lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    """
    The original Lesk algorithm (1986) [1]

    Code From NLTK (https://www.nltk.org/_modules/nltk/wsd.html)

    Return a synset for an ambiguous word in a context.

    :param iter context_sentence: The context sentence.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.

    Example::

        >>> lesk("ดวงจันทร์ ขึ้น ตอน กลางคืน".split(),"ดวงจันทร์")
        Synset('moon.n.06')
        >>> lesk("เมื่อ วันจันทร์ ที่ ผ่านมา".split(),"วันจันทร์")
        Synset('monday.n.01')

    [1] Lesk, Michael. "Automatic sense disambiguation using machine
    readable dictionaries: how to tell a pine cone from an ice cream
    cone." Proceedings of the 5th Annual International Conference on
    Systems Documentation. ACM, 1986.
    http://dl.acm.org/citation.cfm?id=318728
    """

    context = set(context_sentence)
    if synsets is None:
        synsets = wordnet.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense