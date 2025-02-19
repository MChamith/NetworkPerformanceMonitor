ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index):
    """returns one-hot vector with given size and value 1 at given index
    """
    vec = [0 for _ in range(NUM_LETTERS)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    """returns one-hot representation of given letter
    """
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):
    """returns a list of character indices
    Args:
        word: string

    Return:
        indices: int list with length len(word)
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices