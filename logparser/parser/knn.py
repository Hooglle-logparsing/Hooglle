def split_(content):
    result = []
    tmp_word = ""
    for ch in content:
        if (ch.isalnum() or ch in ['.', '-', '_', '/']):
            tmp_word += ch
        else:
            if (tmp_word):
                result.append(tmp_word)
                tmp_word = ""
            result.append(ch)
            if (result[-1] == ">" and result[-2] == "*" and result[-3] == "<"):
                result = result[:-3]
                result.append("<*>")
    if (tmp_word):
        result.append(tmp_word)
    return result


class Invert_Index:
    def __init__(self):
        self.word_table = {}
        self.id_table = {}

    def insert_template(self, template, tid):
        templateL = split_(template)
        count = 0
        for token in templateL:
            if (token.isalpha()):
                count += 1
                if (token not in self.word_table.keys()):
                    self.word_table[token] = {}
                if (tid not in self.word_table[token].keys()):
                    self.word_table[token][tid] = 0
                self.word_table[token][tid] += 1
        self.id_table[tid] = count

    def query(self, template, k):
        templateL = split_(template)
        result = {}
        for token in templateL:
            if (token.isalpha()):
                if (token in self.word_table.keys()):
                    for tid in self.word_table[token]:
                        if (tid not in result.keys()):
                            result[tid] = {'value': 0, 'contents': {}}
                        if(token not in result[tid]['contents'].keys()):
                            result[tid]['contents'][token]=0
                        if (result[tid]['contents'][token] < self.word_table[token][tid]):
                            result[tid]['value'] += 1
                            result[tid]['contents'][token] += 1

        ret = {}
        for tid in result.keys():
            ret[tid] = (result[tid]['value'] * 1.0) / (self.id_table[tid] * 1.0)
        ret = sorted(ret.items(), key=lambda x: x[1], reverse=True)
        return ret[:k]
