import re


class prefixTreeNode:
    def __init__(self, word, cid):
        self.word = word
        self.next = {}
        self.content = ""
        self.contents = []
        self.cid = cid
        self.last = []
        self.wild = []

    def copy(self):
        newNode = prefixTreeNode(self.word, self.cid)
        newNode.next = self.next
        newNode.content = self.content
        newNode.contents = self.contents
        newNode.last = self.last
        newNode.wild = self.wild
        return newNode


def check_characters(content, character_types):
    for c in content:
        if (c.isdigit()):
            character_types.add('digit')
        elif (c.isalpha()):
            character_types.add('alpha')
        else:
            character_types.add(c)
    return character_types


def preprocess(template, wildcards, wild_contents):
    if (template[:3] == "<*>" and " " in wildcards[0]):
        index = wild_contents[0].find(" ")
        wildcards1 = wild_contents[0][:index]
        wildcards2 = wild_contents[0][index + 1:]
        wild1 = check_characters(wildcards1, set())
        wild2 = check_characters(wildcards2, set())
        template = "<*> " + template
        new_wildcards = [wild1, wild2]
        for i in range(1, len(wildcards)):
            new_wildcards.append(wildcards[i])
        return template, new_wildcards
    else:
        return template, wildcards


def Str2List(nodeStr):
    StrList = []
    index = 0
    while (index < len(nodeStr)):
        word_now = nodeStr[index]
        if (word_now != '<'):
            StrList.append(word_now)
            index += 1
        elif (nodeStr[index:index + 3] == "<*>"):
            StrList.append("<*>")
            index = index + 3
        elif (nodeStr[index:index + 2] == "<*"):
            StrList.append("<*>")
            index = index + 2
        else:
            StrList.append("<")
            index += 1
    return StrList


class prefixTree:
    def __init__(self):
        self.root = prefixTreeNode("begin", -1)
        self.cid2node = {}

    def add_wildcards(self, StrList, wildcards):
        node_now = self.root
        wildcard_num = 0
        Str_num = 0
        while (node_now.content != "success" and node_now.next.keys() and wildcard_num < len(wildcards)):
            count = 0
            if (node_now.word == "<*>"):
                count += 1
            for content in node_now.contents:
                if (content == StrList[Str_num]):
                    Str_num += 1
                else:
                    print("wrong")
                    return "wrong"
                if (content == "<*>"):
                    count += 1
            wild_num = 0
            for i in range(count):
                if (wild_num < len(node_now.wild)):
                    node_now.wild[wild_num] = node_now.wild[wild_num].union(wildcards[wildcard_num])
                else:
                    node_now.wild.append(wildcards[wildcard_num])
                wildcard_num += 1
                wild_num += 1
            if (Str_num >= len(StrList) and "success" in node_now.next.keys()):
                break
            node_now = node_now.next[StrList[Str_num]]
            Str_num += 1
        return

    def add_prefix_tree_with_templateTree_with_compress(self, template, cid, wildcards, wild_content, logmessage):
        if (len(re.findall('[^<>* ]', template)) == 0):
            template = logmessage
            wildcards = []
            wild_content = []
        template, wildcards = preprocess(template, wildcards, wild_content)
        StrList = Str2List(template)
        node_now = self.root
        node_now = self.addNodeStrList2PrefixTree(node_now, StrList, cid)
        newNode = prefixTreeNode("success", cid)
        node_now.next["success"] = newNode
        newNode.last.append(node_now)
        self.cid2node[cid] = newNode
        self.add_wildcards(StrList, wildcards)
        return

    def addNodeStrList2PrefixTree(self, node_now, template, cid):
        index = 0
        while (index < len(template)):
            word_now = template[index]
            if (word_now in node_now.next.keys()):
                node_now = node_now.next[word_now]
                index += 1
                if (len(node_now.contents) > 0):
                    index, node_now, i = self.MatchStrWithPrefixTree(node_now, index, template, cid)
                    if (i != -1):
                        newNode = prefixTreeNode(node_now.contents[i], cid)
                        newNode.contents = node_now.contents[i + 1:]
                        for ch in newNode.contents:
                            newNode.content = newNode.content + ch
                        newNode.cid = node_now.cid
                        newNode.next = node_now.next
                        for key in newNode.next.keys():
                            node_next = newNode.next[key]
                            if (node_now in node_next.last):
                                node_next.last.remove(node_now)
                                node_next.last.append(newNode)

                        node_now.contents = node_now.contents[:i]
                        node_now.content = ""
                        if (node_now.cid != cid):
                            node_now.cid = -1
                        for ch in node_now.contents:
                            node_now.content = node_now.content + ch
                        node_now.next = {}
                        node_now.next[newNode.word] = newNode
                        newNode.last.append(node_now)

                        count = 0
                        if (node_now.word == "<*>"):
                            count += 1
                        for c in node_now.contents:
                            if (c == "<*>"):
                                count += 1
                        newNode.wild = node_now.wild[count:]
                        node_now.wild = node_now.wild[:count]
                else:
                    if (node_now.cid != cid):
                        node_now.cid = -1
            else:
                newNode = prefixTreeNode(word_now, cid)
                newNode.contents = template[index + 1:]
                for ch in newNode.contents:
                    newNode.content = newNode.content + ch
                node_now.next[word_now] = newNode
                if (node_now.cid != cid):
                    node_now.cid = -1
                count = 0
                if (node_now.word == "<*>"):
                    count += 1
                for c in node_now.contents:
                    if (c == "<*>"):
                        count += 1
                newNode.wild = node_now.wild[count:]
                node_now.wild = node_now.wild[:count]
                newNode.last.append(node_now)
                node_now = node_now.next[word_now]
                break
        return node_now

    def MatchStrWithPrefixTree(self, node_now, index, nodeStrList, cid):
        isBreak = False
        for i in range(len(node_now.contents)):
            if (index >= len(nodeStrList)):
                return index, node_now, i
            word = nodeStrList[index]
            ch = node_now.contents[i]
            if (word == ch):
                index += 1
            else:
                newNode = prefixTreeNode(ch, cid)
                newNode.next = node_now.next
                newNode.contents = node_now.contents[i + 1:]
                newNode.cid = node_now.cid
                for c in newNode.contents:
                    newNode.content = newNode.content + c
                for key in newNode.next.keys():
                    node_next = newNode.next[key]
                    if (node_now in node_next.last):
                        node_next.last.remove(node_now)
                        node_next.last.append(newNode)
                node_now.next = {}
                node_now.next[ch] = newNode
                newNode.last.append(node_now)

                if (node_now.cid != cid):
                    node_now.cid = -1
                node_now.contents = node_now.contents[:i]
                node_now.content = ""
                for c in node_now.contents:
                    node_now.content = node_now.content + c
                isBreak = True

                count = 0
                if (node_now.word == "<*>"):
                    count += 1
                for c in node_now.contents:
                    if (c == "<*>"):
                        count += 1
                newNode.wild = node_now.wild[count:]
                node_now.wild = node_now.wild[:count]

                break
        if (not isBreak and node_now.cid != cid):
            node_now.cid = -1
        return index, node_now, -1

    def delete_prefix_tree(self, cid):
        nodeStack = []
        cidNode = self.cid2node[cid]
        nodeStack.append(cidNode)
        while (nodeStack):
            node_now = nodeStack.pop()
            if (node_now.cid == cid):
                for node_last in node_now.last:
                    if (node_now.word in node_last.next.keys()):
                        node_last.next.pop(node_now.word)
                        nodeStack.append(node_last)
                    if (len(node_last.next) == 1 and node_last.cid == -1 and node_last.word != "begin"):
                        key = ""
                        for k in node_last.next.keys():
                            key = k
                        if (node_last.next[key].word == "success"):
                            continue
                        if (node_last.next[key].cid != cid):
                            tmp = node_last.next[key]
                            node_last.contents.append(tmp.word)
                            node_last.contents = node_last.contents + tmp.contents
                            node_last.content = node_last.content + tmp.word + tmp.content
                            node_last.cid = tmp.cid
                            node_last.next = tmp.next
                            for key in node_last.next.keys():
                                nextNode = node_last.next[key]
                                nextNode.last.remove(tmp)
                                nextNode.last.append(node_last)
                del node_now
        self.cid2node.pop(cid)
        return

    def match(self, log, node_now):
        index = 0
        while (index < len(log)):
            if (node_now.content != ""):
                index, match_index = match_Str(log, index, node_now.contents, node_now.wild,
                                               next_chs=node_now.next.keys())
                if (index == -1):
                    return -1
                if (index >= len(log)):
                    if ("success" in node_now.next.keys() and match_index >= len(
                            node_now.contents)):
                        return node_now.next["success"].cid
                    else:
                        return -1
                ch_now = log[index]
                if (ch_now in node_now.next.keys()):
                    node_tmp = node_now.next[ch_now]
                    # if (len(node_tmp.contents) != 0):
                    index_ = index + 1
                    cid = self.match(log[index_:], node_tmp)
                    if (cid != -1):
                        return cid
                if ('<*>' in node_now.next.keys()):
                    node_tmp = node_now.next["<*>"]
                    if (len(node_tmp.contents) == 0 and "success" in node_tmp.next.keys() and not Contain_others(
                            log[index:], node_tmp.wild[0])):
                        return node_tmp.next["success"].cid
                    elif (len(node_tmp.contents) != 0):
                        if (len(node_tmp.contents) > 0):
                            wilds = node_tmp.wild[0]
                            index_ = match_wildcard(log, index, wilds, [node_tmp.contents[0]])
                            node_tmp = node_tmp.copy()
                            node_tmp.wild = node_tmp.wild[1:]
                            for idx in index_:
                                cid = self.match(log[idx:], node_tmp)
                                if (cid != -1):
                                    return cid
                    else:
                        values = node_tmp.next.keys()
                        wilds = node_tmp.wild[0]
                        index_ = match_wildcard(log, index, wilds, values)
                        node_tmp = node_tmp.copy()
                        node_tmp.wild = node_tmp.wild[1:]
                        for idx in index_:
                            cid = self.match(log[idx:], node_tmp)
                            if (cid != -1):
                                return cid
                return -1
            else:
                ch_now = log[index]
                if (ch_now in node_now.next.keys()):
                    node_tmp = node_now.next[ch_now]
                    index_ = index + 1
                    cid = self.match(log[index_:], node_tmp)
                    if (cid != -1):
                        return cid
                if ('<*>' in node_now.next.keys()):
                    node_tmp = node_now.next["<*>"]
                    if (len(node_tmp.contents) == 0 and "success" in node_tmp.next.keys() and not Contain_others(
                            log[index:], node_tmp.wild[0])):
                        return node_tmp.next["success"].cid
                    elif (len(node_tmp.contents) != 0):
                        if (len(node_tmp.contents) > 0):
                            wilds = node_tmp.wild[0]
                            index_ = match_wildcard(log, index, wilds, [node_tmp.contents[0]])
                            node_tmp = node_tmp.copy()
                            node_tmp.wild = node_tmp.wild[1:]
                            for idx in index_:
                                cid = self.match(log[idx:], node_tmp)
                                if (cid != -1):
                                    return cid

                    else:
                        values = node_tmp.next.keys()
                        wilds = node_tmp.wild[0]
                        index_ = match_wildcard(log, index, wilds, values)
                        node_tmp = node_tmp.copy()
                        node_tmp.wild = node_tmp.wild[1:]
                        for idx in index_:
                            cid = self.match(log[idx:], node_tmp)
                            if (cid != -1):
                                return cid
                return -1

        if (len(node_now.contents) > 0 and log == ""):
            return -1
        if ("success" in node_now.next.keys()):
            return node_now.next["success"].cid
        else:
            return -1


def Contain_others(log, wilds):
    containDigit = False
    containAlpha = False
    if ('digit' in wilds):
        containDigit = True
    if ('alpha' in wilds):
        containAlpha = True
    for ch_now in log:
        if (ch_now.isdigit() and not containDigit):
            return True
        if (ch_now.isalpha() and not containAlpha):
            return True
        if (not ch_now.isalnum() and ch_now not in wilds):
            return True
    return False


def match_wildcard(log, index, wilds, stopdelimiter):
    containDigit = False
    containAlpha = False
    if ('digit' in wilds):
        containDigit = True
    if ('alpha' in wilds):
        containAlpha = True
    indexs = []
    while (index < len(log)):
        ch_now = log[index]
        if (ch_now.isdigit() and not containDigit):
            indexs.append(index)
            return indexs
        if (ch_now.isalpha() and not containAlpha):
            indexs.append(index)
            return indexs
        if (not ch_now.isalnum() and ch_now not in wilds):
            indexs.append(index)
            return indexs
        if (ch_now not in stopdelimiter):
            index += 1
        else:
            indexs.append(index)
            index += 1
    if (not indexs):
        indexs = [len(log)]
    return indexs


def match_Str(log, index, StringL, wild, next_chs=[]):
    match_index = 0
    wild_index = 0
    ch_index = index
    while (ch_index < len(log)):
        if (match_index >= len(StringL)):
            return ch_index, match_index
        ch_now = log[ch_index]
        match_now = StringL[match_index]
        if (match_now == "<*>"):
            wilds = wild[wild_index]
            wild_index += 1
            if (match_index + 1 >= len(StringL)):
                match_next = []
                for ch in next_chs:
                    if (ch != "success"):
                        match_next.append(ch)

            else:
                match_next = [StringL[match_index + 1]]

            indexs = match_wildcard(log, ch_index, wilds, match_next)
            match_index += 1
            tmp = match_index
            for i in indexs:
                ch_index, match_index = match_Str(log, i, StringL[tmp:], wild[1:], next_chs)
                if (ch_index != -1):
                    break
            if (ch_index == -1):
                return -1, -1
            else:
                return ch_index, match_index + tmp


        else:
            if (ch_now != match_now):
                return -1, -1
            else:
                ch_index += 1
                match_index += 1
    return len(log), match_index


def lcs(strL1_o, strL2_o):
    strL1 = strL1_o.copy()
    strL2 = strL2_o.copy()
    strL1.reverse()
    strL2.reverse()
    len1 = len(strL1)
    len2 = len(strL2)

    dp = [[0 for column in range(len2 + 1)] for row in range(len1 + 1)]
    trace_back = [["None" for column in range(len2 + 1)] for row in range(len1 + 1)]

    for i in range(len(dp)):
        trace_back[i][0] = 'up'
    for i in range(len(dp[0])):
        trace_back[0][i] = 'left'
    trace_back[0][0] = 'start'

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if (strL1[i - 1] == strL2[j - 1]):
                dp[i][j] = dp[i - 1][j - 1] + 1
                trace_back[i][j] = 'diag'
            else:
                if (dp[i - 1][j] >= dp[i][j - 1]):
                    dp[i][j] = dp[i - 1][j]
                    trace_back[i][j] = 'up'
                else:
                    dp[i][j] = dp[i][j - 1]
                    trace_back[i][j] = 'left'
    pairs = []
    i_now = len1
    j_now = len2
    while (trace_back[i_now][j_now] != 'start'):
        if (trace_back[i_now][j_now] == 'diag'):
            pairs.append([i_now - 1, j_now - 1])
            i_now -= 1
            j_now -= 1
        elif (trace_back[i_now][j_now] == 'up'):
            i_now -= 1
        else:
            j_now -= 1
    new_pairs = []
    for pair in pairs:
        new_pairs.append([len1 - 1 - pair[0], len2 - 1 - pair[1]])
    return new_pairs


def Heuristic_parse(log):
    wildcards = []
    wilds = []
    tmpL = log.split(" ")
    result = []
    for t in tmpL:
        if (re.findall('\d', t)):
            result.append("<*>")
            character_types = set()
            character_types = check_characters(t, character_types)
            wildcards.append(character_types)
            wilds.append(t)
        else:
            result.append(t)
    ret = " ".join(result)
    if (len(re.findall("[^<>* ]", ret)) == 0):
        ret = ""
        for ch in log:
            if (ch.isalnum()):
                if (len(ret) > 3 and ret[-3:] == "<*>"):
                    continue
                ret += "<*>"
            else:
                ret += ch
    ret, wildcards, wilds = match_wildcard_with_content(ret, log)
    return ret, wildcards, wilds


def match_wildcard_with_content(template, log):
    templateL = Str2List(template + "※")
    logL = Str2List(log + "※")
    alignPairL = lcs(logL, templateL)
    T_index = 0
    L_index = 0
    A_index = 0
    template = ""
    wildcards = []
    wild_content = []
    while (T_index < len(templateL) and L_index < len(logL) and A_index < len(alignPairL)):
        pair = alignPairL[A_index]
        if (T_index == pair[1] and L_index == pair[0]):
            if (templateL[T_index] == "<*>"):
                wildcards.append(check_characters(logL[L_index], set()))
                wild_content.append(logL[L_index])
            template += templateL[T_index]
            T_index += 1
            L_index += 1
            A_index += 1
            continue

        if (templateL[T_index] == "<*>" and alignPairL[A_index][1] - T_index == 1):
            if (logL[L_index] == " "):
                template += " "
                L_index += 1
            character_types = set()
            for i in range(L_index, pair[0]):
                character_types = check_characters(logL[i], character_types)

            if ('digit' not in character_types and 'alpha' not in character_types):
                for i in range(L_index, pair[0]):
                    template += logL[i]
            else:
                tmp = ""
                for i in range(L_index, pair[0]):
                    tmp += logL[i]

                if (tmp[0] in ["{", "[", "("] and tmp[-1] in ["}", "]", ")"]):
                    template += tmp[0] + "<*>" + tmp[-1]
                    tmp = tmp[1:-1]
                else:
                    template += "<*>"

                character_types = set()
                character_types = check_characters(tmp, character_types)
                wildcards.append(character_types)
                wild_content.append(tmp)

        elif ('<*>' not in templateL[T_index:pair[1]] and template[-3:] != "<*>"):
            for i in range(L_index, pair[0]):
                template += logL[i]
        elif (template[-3:] == "<*>"):
            tmp = wild_content[-1]
            for i in range(L_index, pair[0]):
                tmp += logL[i]
            wildcards[-1] = check_characters(tmp, set())
            wild_content[-1] = tmp
        else:
            character_types = set()
            for i in range(L_index, pair[0]):
                character_types = check_characters(logL[i], character_types)
            if (character_types):
                if ('digit' not in character_types and 'alpha' not in character_types):
                    for i in range(L_index, pair[0]):
                        template += logL[i]
                else:
                    template += "<*>"
                    wildcards.append(character_types)
                    tmp = ""
                    for i in range(L_index, pair[0]):
                        tmp += logL[i]
                    wild_content.append(tmp)
        T_index = pair[1]
        L_index = pair[0]

    return template.replace("※", ""), wildcards, wild_content


def Heuristic_check(template, wildcards, wild_content):
    templateL = Str2List2(template)
    template_new = ""
    wildcards_new = []
    wild_content_new = []
    idx_wild = 0
    for ch in templateL:
        if (ch != "<*>"):
            template_new += ch
        else:
            if (len(re.findall(" ", wild_content[idx_wild])) > 2):
                w = []
                w_c = []
                tmpL = wild_content[idx_wild].split(" ")
                result = []
                for t in tmpL:
                    if (re.findall('\d', t)):
                        result.append("<*>")
                        character_types = set()
                        character_types = check_characters(t, character_types)
                        wildcards.append(character_types)
                        w_c.append(t)
                    else:
                        result.append(t)
                ret = " ".join(result)
                template_new += ret
                for ww in w:
                    wildcards_new.append(ww)
                for ww in w_c:
                    wild_content_new.append(ww)
            else:
                template_new += ch
                wildcards_new.append(wildcards[idx_wild])
                wild_content_new.append(wild_content[idx_wild])
            idx_wild += 1
    return template_new, wildcards_new, wild_content_new


def Str2List2(nodeStr):
    StrList = []
    index = 0
    tmp = ""
    while (index < len(nodeStr)):
        word_now = nodeStr[index]
        if (word_now != '<'):
            if (word_now.isalnum()):
                tmp += word_now
            else:
                if (tmp):
                    StrList.append(tmp)
                    tmp = ""
                StrList.append(word_now)
            index += 1
        elif (nodeStr[index:index + 3] == "<*>"):
            if (tmp):
                StrList.append(tmp)
                tmp = ""
            StrList.append("<*>")
            index = index + 3
        elif (nodeStr[index:index + 2] == "<*"):
            if (tmp):
                StrList.append(tmp)
                tmp = ""
            StrList.append("<*>")
            index = index + 2
        else:
            if (tmp):
                StrList.append(tmp)
                tmp = ""
            StrList.append("<")
            index += 1
    if (tmp):
        StrList.append(tmp)
    return StrList


def merge_two_template(template1, template2):
    template1L = Str2List2(template1)
    template2L = Str2List2(template2)
    alignPairL = lcs(template1L, template2L)
    T1_index = 0
    T2_index = 0
    A_index = 0
    wilds = []
    template = ""
    unmatch = 0
    AVG_LEN = (len(template1L) + len(template2L)) / 2
    while (T1_index < len(template1L) and T2_index < len(template2L) and A_index < len(alignPairL)):
        pair = alignPairL[A_index]
        if (T1_index == pair[0] and T2_index == pair[1]):
            template += template1L[T1_index]
            if (template[-6:] == "<*><*>"):
                template = template[:-3]
                wilds[-1][0] = wilds[-1][0] + "<*>"
                wilds[-1][1] = wilds[-1][1] + "<*>"
            T1_index += 1
            T2_index += 1
            A_index += 1
            continue
        else:
            template += "<*>"
            tmp1 = ""
            for i in range(T1_index, pair[0]):
                tmp1 += template1L[i]
            tmp2 = ""
            for i in range(T2_index, pair[1]):
                tmp2 += template2L[i]
            if (template[-6:] == "<*><*>"):
                template = template[:-3]
                tmp1 = "<*>" + tmp1
                tmp2 = "<*>" + tmp2
            if (len(template) > 4 and template[-3:] == "<*>" and template[-4].isalnum()):
                if (tmp1 and not tmp1[0].isalnum() and tmp1[0] not in "<*>"):
                    template = template[:-3] + tmp1[0] + "<*>"
                    if (tmp2 and tmp2[0] == tmp1[0]):
                        tmp2 = tmp2[1:]
                    tmp1 = tmp1[1:]
                elif (tmp2 and not tmp2[0].isalnum() and tmp2[0] not in "<*>"):
                    template = template[:-3] + tmp2[0] + "<*>"
                    if (tmp1 and tmp1[0] == tmp2[0]):
                        tmp1 = tmp1[1:]
                    tmp2 = tmp2[1:]
            wilds.append([tmp1, tmp2])
            unmatch += max(pair[0] - T1_index, pair[1] - T2_index)
            T1_index = pair[0]
            T2_index = pair[1]
    if (T1_index < len(template1L) or T2_index < len(template2L) or A_index < len(alignPairL)):
        template += "<*>"
        unmatch += max(len(template1L) - T1_index, len(template2L) - T2_index)
        tmp1 = ""
        for i in range(T1_index, len(template1L)):
            tmp1 += template1L[i]
        tmp2 = ""
        for i in range(T2_index, len(template2L)):
            tmp2 += template2L[i]
        wilds.append([tmp1, tmp2])
    new_wild_rate = unmatch * 1.0 / AVG_LEN
    return template, wilds, new_wild_rate
