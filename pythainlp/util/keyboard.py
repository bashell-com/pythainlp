# -*- coding: utf-8 -*-
"""
Correct text in one language that is incorrectly-typed
with a keyboard layout in another language.
"""
EN_TH_KEYB_PAIRS = {
    "Z": "(",
    "z": "ผ",
    "X": ")",
    "x": "ป",
    "C": "ฉ",
    "c": "แ",
    "V": "ฮ",
    "v": "อ",
    "B": "ฺ",
    "b": "ิ",
    "N": "์",
    "n": "ื",
    "M": "?",
    "m": "ท",
    "<": "ฒ",
    ",": "ม",
    ">": "ฬ",
    ".": "ใ",
    "?": "ฦ",
    "/": "ฝ",
    "A": "ฤ",
    "a": "ฟ",
    "S": "ฆ",
    "s": "ห",
    "D": "ฏ",
    "d": "ก",
    "F": "โ",
    "f": "ด",
    "G": "ฌ",
    "g": "เ",
    "H": "็",
    "h": "้",
    "J": "๋",
    "j": "่",
    "K": "ษ",
    "k": "า",
    "L": "ศ",
    "l": "ส",
    ":": "ซ",
    ";": "ว",
    '"': ".",
    "'": "ง",
    "Q": "๐",
    "q": "ๆ",
    "W": '"',
    "w": "ไ",
    "E": "ฎ",
    "e": "ำ",
    "R": "ฑ",
    "r": "พ",
    "T": "ธ",
    "t": "ะ",
    "Y": "ํ",
    "y": "ั",
    "U": "๊",
    "u": "ี",
    "I": "ณ",
    "i": "ร",
    "O": "ฯ",
    "o": "น",
    "P": "ญ",
    "p": "ย",
    "{": "ฐ",
    "[": "บ",
    "}": ",",
    "]": "ล",
    "|": "ฅ",
    "\\": "ฃ",
    "~": "%",
    "`": "_",
    "@": "๑",
    "2": "/",
    "#": "๒",
    "3": "-",
    "$": "๓",
    "4": "ภ",
    "%": "๔",
    "5": "ถ",
    "^": "ู",
    "6": "ุ",
    "&": "฿",
    "7": "ึ",
    "*": "๕",
    "8": "ค",
    "(": "๖",
    "9": "ต",
    ")": "๗",
    "0": "จ",
    "_": "๘",
    "-": "ข",
    "+": "๙",
    "=": "ช",
}

TH_EN_KEYB_PAIRS = {v: k for k, v in EN_TH_KEYB_PAIRS.items()}

EN_TH_TRANSLATE_TABLE = str.maketrans(EN_TH_KEYB_PAIRS)
TH_EN_TRANSLATE_TABLE = str.maketrans(TH_EN_KEYB_PAIRS)


def eng_to_thai(text: str) -> str:
    """
    Corrects the given text that was incorrectly typed using English-US
    Qwerty keyboard layout to the originally intended keyboard layout
    that is the Thai Kedmanee keyboard.

    :param str text: incorrect text input (type Thai with English keyboard)
    :return: Thai text where incorrect typing with
             a keyboard layout is corrected
    :rtype: str

    :Example:

        Intentionally type "ธนาคารแห่งประเทศไทย", but got "Tok8kicsj'xitgmLwmp"

        >>> from pythainlp.util import eng_to_thai
        >>>
        >>> eng_to_thai("Tok8kicsj'xitgmLwmp")
        ธนาคารแห่งประเทศไทย
    """
    return text.translate(EN_TH_TRANSLATE_TABLE)


def thai_to_eng(text: str) -> str:
    """
    Corrects the given text that was incorrectly typed using Thai Kedmanee
    keyboard layout to the originally intended keyboard layout
    that is the English-US Qwerty keyboard.

    :param str text: incorrect text input (type English with Thai keyboard)
    :return: English text where incorrect typing with
             a keyboard layout is corrected
    :rtype: str

    :Example:

        Intentionally type "Bank of Thailand", but got "ฺฟืา นด ธ้ฟรสฟืก".

        >>> from pythainlp.util import eng_to_thai
        >>>
        >>> thai_to_eng("ฺฟืา นด ธ้ฟรสฟืก")
        'Bank of Thailand'
    """
    return text.translate(TH_EN_TRANSLATE_TABLE)
