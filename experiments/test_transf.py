import string


def strip_chars(inpstr, to_strip=string.punctuation):
    strar = list(inpstr)
    stripped_away_front = ""
    stripped_away_back = ""

    for i in reversed(range(0, len(strar))):
        if strar[i] in to_strip:
            stripped_away_back += strar[i]
            del strar[i]
        else:
            break
    lcount = 0
    while lcount < len(strar):
        if strar[lcount] in to_strip:
            stripped_away_front += strar[lcount]
            del strar[lcount]
            lcount -= 1
        else:
            break
        lcount += 1

    return stripped_away_front, ''.join(strar), stripped_away_back[::-1]


def remove_possessive(st):
    return st if len(st) == 1 else (st[:-2] if st.rfind("'s") == len(st) - 2 else st)


if __name__ == '__main__':
    print(strip_chars('.>?>?>/asdfasdfasdf>..?>?>?>?'))
    print(strip_chars('I'))

    print('----------')

    print(remove_possessive('a'))