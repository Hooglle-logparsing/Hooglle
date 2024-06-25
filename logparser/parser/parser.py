from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from logparser.parser.prefix_tree import prefixTree, match_wildcard_with_content, merge_two_template, Heuristic_parse, \
    Str2List2
import pandas as pd
import os
import re
import json
import scipy.special
from logparser.parser.knn import Invert_Index
from langchain import PromptTemplate, LLMChain


class OpenLLMAPI(LLM):
    import openai
    client: openai.Client
    model: str

    @property
    def _llm_type(self) -> str:
        return "OpenLLMAPI"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs
    ) -> str:
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = 512
        if 'n' in kwargs and kwargs['n'] != 1:
            kwargs['n'] = 1
            print('Warning: resetting n=1')
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            stop=stop,
            temperature=0,
            **kwargs,
        )
        result = result.choices[0].message.content.strip()
        return result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"client": self.client, "model": self.model}


def create_open_llm(url):
    if (not url):
        return None
    try:
        import openai
        client = openai.Client(
            api_key="empty",
            base_url=url)
        models = client.models.list()
        model = models.data[0].id
        ret = OpenLLMAPI(client=client,model=model)
        response = ret.predict('hello')
        print('url:', url)
        print('model:', model)
        return ret
    except:
        print("LLM connect error!")
        return None


def parse_result(text):
    if ('Final Answer:' in text):
        index = text.find('Final Answer:')
        text = text[index + len('Final Answer:'):]
    return text


class LogCluster:
    def __init__(self, logIDL=[], template=""):
        self.template = template
        self.logIDL = logIDL
        self.logs = []


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/'):
        self.path = indir
        self.savePath = outdir
        self.log_format = log_format
        self.prefix_tree = prefixTree()
        self.logClusters = []

    def log_to_dataframe(self, log_file, regex, headers):
        log_messages = []
        linecount = 0
        with open(log_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers)

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def parse_log_with_LLM(self, logmessage):
        try:
            resultString = self.llm.predict(logmessage)
        except:
            resultString, wildcards, _ = Heuristic_parse(logmessage)
        self.Count_Call_LLM += 1
        if ('<*>' not in resultString):
            resultString = logmessage
        return resultString

    def outputResults(self):
        print("Parsing done, outputing results. Call LLM for " + str(self.Count_Call_LLM) + " times.")

        filename = self.logName
        df_event = []
        ids = [-1] * self.df_log.shape[0]
        templates = [""] * self.df_log.shape[0]

        for cid in range(len(self.logClusters)):
            cluster = self.logClusters[cid]
            df_event.append([cid, cluster.template, len(cluster.logIDL)])

            for id in cluster.logIDL:
                ids[id] = cid
                templates[id] = cluster.template

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        self.df_log.to_csv(os.path.join(self.savePath, filename + '_structured.csv'), index=False,
                           encoding="utf-8")
        df_event.to_csv(os.path.join(self.savePath, filename + '_templates.csv'), index=False, encoding="utf-8")

    def llm_result_parse(self, result):
        result_lower = result.lower()
        ret = result_lower.split('Result')[-1]
        if ('yes' in ret):
            return True
        else:
            return False

    def llm_result_parse2(self, result):
        result_lower = result.lower()
        if ('constant' in result_lower):
            return True
        else:
            return False

    def correct_ask(self, template1, template2):
        result = self.merge_two_determine(template1, template2)
        return self.llm_result_parse(result)

    def v_c_determination(self, content, log):
        result = self.V_C_CoT(log, content)
        return self.llm_result_parse2(result)

    def cover(self, wilds):
        for w1, w2 in wilds:
            w1TF = True
            w2TF = True
            if ("<*>" not in w1 or len(re.findall("[a-zA-Z0-9]", w1)) > 0):
                w1TF = False
            if ("<*>" not in w2 or len(re.findall("[a-zA-Z0-9]", w2)) > 0):
                w2TF = False
            if (not w1TF and not w2TF):
                return False
        return True

    def digit(self, wilds):
        for w1, w2 in wilds:
            if (not (w1.isdigit() and w2.isdigit())):
                return False
        return True

    def process_wilds(self, template):
        while ("<*> <*>" in template):
            template = re.sub("<\*> <\*>", "<*>", template)
        return template

    def constant_refinement(self, template, log):
        templateL = Str2List2(template)
        changed = []
        for content in templateL:
            if (not content.isdigit() and not (
                    re.fullmatch(r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b", content) and len(
                re.findall("\d", content)) > 0)):
                continue
            if (not self.v_c_determination(content, log)):
                changed.append(content)
        while (changed):
            change_new = []
            not_changed = []
            for c in changed:
                add = True
                for h in changed:
                    if (c == h):
                        continue
                    if (c in h):
                        add = False
                        break
                if (add):
                    change_new.append(c)
                else:
                    not_changed.append(c)
            for content in change_new:
                template = re.sub(content, '<*>', template)
            changed = not_changed

        template, wildcards, wild_contents = match_wildcard_with_content(template, log)
        if (len(re.findall('[a-zA-Z0-9]', template)) == 0):
            template, wildcards, _ = Heuristic_parse(log)
        return template, wildcards

    def variable_refinement(self, parsed_log, logmessage):
        template, wildcards, wild_contents = match_wildcard_with_content(parsed_log, logmessage)
        wild_c = []
        for content in wild_contents:
            if (not content.isalpha() or len(content) < 3 or content.lower() in ['true', 'false', 'null']
                    or re.fullmatch("\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b", content)):
                wild_c.append("<*>")
                continue
            if (not self.v_c_determination(content, logmessage)):
                wild_c.append("<*>")
            else:
                wild_c.append(content)
        templateL = Str2List2(template)
        index_now = 0
        for i in range(len(wild_c)):
            index = index_now
            for j in range(len(templateL)):
                if (templateL[j] == "<*>" and index > 0):
                    index -= 1
                elif (templateL[j] == "<*>" and index == 0):
                    templateL[j] = wild_c[i]
                    if (wild_c[i] == "<*>"):
                        index_now += 1
                    break
        template = ""
        for t in templateL:
            template += t
        template, wildcards, wild_contents = match_wildcard_with_content(template, logmessage)
        if (len(re.findall('[a-zA-Z0-9]', template)) == 0):
            template, wildcards, _ = Heuristic_parse(logmessage)
        return template, wildcards

    def V_C_CoT(self, log, content):
        from logparser.parser.prompts import CONTENT_INQUIRY, CONSTANT_VARIABLE_INQUIRY
        prompt = PromptTemplate(template=CONTENT_INQUIRY, input_variables=["log", "content"])
        llm_chain1 = LLMChain(prompt=prompt, llm=self.llm2)
        response = llm_chain1.run(log=log, content=content)

        prompt2 = PromptTemplate(template=CONSTANT_VARIABLE_INQUIRY, input_variables=["analyse", "log", "content"])
        llm_chain2 = LLMChain(prompt=prompt2, llm=self.llm2)
        response2 = llm_chain2.run(analyse=response, log=log, content=content)
        if ("constant" in response2):
            return "constant"
        else:
            return "variable"

    def merge_two_determine(self, template1, template2):
        from logparser.parser.prompts import MERGE_TWO_INQUIRY
        prompt = PromptTemplate(template=MERGE_TWO_INQUIRY, input_variables=["template1", "template2"])
        llm_chain1 = LLMChain(prompt=prompt, llm=self.llm2)
        response = llm_chain1.run(template1=template1, template2=template2)
        return response

    def merge_same_clusters(self, cid):
        cidL = [cid]
        log_idL = self.logClusters[cid].logIDL.copy()
        logs = self.logClusters[cid].logs.copy()
        for id in range(len(self.logClusters)):
            if (cid == id):
                continue
            template1 = self.logClusters[cid].template
            template2 = self.logClusters[id].template
            merge, template_merge, new_Wilds = self.Quick_judgment(template1, template2)
            if (not merge):
                continue
            if (template_merge == template1 or template_merge == template2):
                cidL.append(id)
                log_idL += self.logClusters[id].logIDL
                logs += self.logClusters[id].logs
                self.logClusters[cid].template = template_merge

        if (len(cidL) == 1):
            return

        new_Clusters = []
        new_prefix_tree = prefixTree()
        new_index = Invert_Index()
        for id in range(len(self.logClusters)):
            if (id not in cidL):
                cid_now = len(new_Clusters)
                new_Clusters.append(self.logClusters[id])
                new_index.insert_template(self.logClusters[id].template, cid_now)
                for log in self.logClusters[id].logs:
                    template, wildcards, wild_content = match_wildcard_with_content(self.logClusters[id].template, log)
                    new_prefix_tree.add_prefix_tree_with_templateTree_with_compress(template, cid_now,
                                                                                    wildcards, wild_content,
                                                                                    log)
        newCluster = LogCluster(log_idL, self.logClusters[cid].template)
        newCluster.logs = logs
        cid_now = len(new_Clusters)
        new_Clusters.append(newCluster)
        new_index.insert_template(newCluster.template, cid_now)
        for log in newCluster.logs:
            template, wildcards, wild_content = match_wildcard_with_content(newCluster.template, log)
            new_prefix_tree.add_prefix_tree_with_templateTree_with_compress(template, cid_now,
                                                                            wildcards, wild_content,
                                                                            log)
        self.logClusters = new_Clusters
        self.prefix_tree = new_prefix_tree
        self.index = new_index

    def parse(self, logName):
        self.logName = logName
        self.load_data()
        self.Count_Call_LLM = 0
        self.index = Invert_Index()
        self.llm = create_open_llm("")
        if (self.llm is None):
            print("Please input the right url of template extraction model")
            return
        self.llm2 = create_open_llm("")
        if (self.llm2 is None):
            print("Please input the right url of template refinement model")
            return

        for idx, line in self.df_log.iterrows():
            if idx % 2000 == 0:
                print('Processed {0:.1f}% of log lines.'.format(idx * 100.0 / len(self.df_log)))
                print(len(self.logClusters))
                print(self.Count_Call_LLM)
            lineID = line['LineId']
            logmessage = line['Content'].strip()
            logid = lineID - 1
            logmessage = re.sub("  ", " ", logmessage)

            match_id = self.prefix_tree.match(logmessage, self.prefix_tree.root)
            if (match_id != -1):
                self.logClusters[match_id].logIDL.append(logid)
                continue
            parsed_log = self.parse_log_with_LLM(logmessage)

            if (len(re.findall("[a-zA-Z0-9]", parsed_log)) == 0 or len(re.findall("<\*>", parsed_log)) > 50):
                parsed_log, wildcards, _ = Heuristic_parse(logmessage)
            index_result = self.index.query(parsed_log, 3)
            matched = False

            template, wildcards = self.variable_refinement(parsed_log, logmessage)
            template, wildcards = self.constant_refinement(template, logmessage)

            for (most_similar_cid, score) in index_result:
                if (score < 0.5):
                    break
                if (most_similar_cid != -1):
                    merge, template_merge, new_Wilds = self.Quick_judgment(self.logClusters[most_similar_cid].template,
                                                                           template)
                    if (merge):
                        if (self.logClusters[most_similar_cid].template == template or self.cover(
                                new_Wilds) or self.digit(new_Wilds)):
                            self.logClusters[most_similar_cid].logIDL.append(logid)
                            self.logClusters[most_similar_cid].logs.append(logmessage)
                            self.logClusters[most_similar_cid].template = template_merge
                            template, wildcards, wild_content = match_wildcard_with_content(template_merge, logmessage)
                            self.prefix_tree.add_prefix_tree_with_templateTree_with_compress(template, most_similar_cid,
                                                                                             wildcards, wild_content,
                                                                                             logmessage)
                            matched = True
                            break
                        if (self.correct_ask(template, self.logClusters[most_similar_cid].template)):
                            self.logClusters[most_similar_cid].logIDL.append(logid)
                            self.logClusters[most_similar_cid].logs.append(logmessage)
                            self.logClusters[most_similar_cid].template = template_merge
                            template, wildcards, wild_content = match_wildcard_with_content(template_merge, logmessage)
                            self.prefix_tree.add_prefix_tree_with_templateTree_with_compress(template, most_similar_cid,
                                                                                             wildcards, wild_content,
                                                                                             logmessage)
                            self.merge_same_clusters(most_similar_cid)
                            matched = True
                            break

            if (not matched):
                cid = len(self.logClusters)
                template, wildcards, wild_content = match_wildcard_with_content(template, logmessage)
                newCluster = LogCluster([logid], template)
                newCluster.logs.append(logmessage)
                self.logClusters.append(newCluster)
                self.prefix_tree.add_prefix_tree_with_templateTree_with_compress(template, cid, wildcards, wild_content,
                                                                                 logmessage)
                self.index.insert_template(template, cid)
                self.merge_same_clusters(cid)
        self.outputResults()
        return

    def Quick_judgment(self, template1, template2):
        template_merge, wilds, new_wild_rate = merge_two_template(template1, template2)
        if (template1 == template2):
            return True, template_merge, wilds
        t1 = template1
        t2 = template2
        while ("<*> <*>" in t1):
            t1 = re.sub("<\*> <\*>", "<*>", t1)
        while ("<*> <*>" in t2):
            t2 = re.sub("<\*> <\*>", "<*>", t2)
        if (t1 == t2):
            return True, t1, []
        if (len(re.findall('[a-zA-Z0-9]', template_merge)) == 0):
            return False, template_merge, wilds
        for wild in wilds:
            for w in wild:
                if (len(re.findall("[a-zA-Z0-9]", w)) == 0 and "<*>" not in w and w != ""):
                    return False, template_merge, wilds
                if (len(re.findall(" ", w)) >= 2 and len(re.findall("[a-zA-Z0-9]", w)) > 0):
                    return False, template_merge, wilds
        if (template1 == template2 or template1 == template_merge or template2 == template_merge or self.cover(wilds)):
            return True, template_merge, wilds
        if (len(re.findall("[a-zA-Z0-9]+", str(wilds))) == 0):
            return False, template_merge, wilds
        if (new_wild_rate > 0.2):
            return False, template_merge, wilds
        if (len(re.findall("<\*> <\*> <\*>", template_merge)) > len(re.findall("<\*> <\*> <\*>", template1)) and len(
                re.findall("<\*> <\*> <\*>", template_merge)) > len(re.findall("<\*> <\*> <\*>", template2))):
            return False, template_merge, wilds
        return True, template_merge, wilds


if __name__ == '__main__':
    benchmark_settings = {
        'HDFS': {
            'log_file': 'HDFS/HDFS_2k.log',
            'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        },

        'Hadoop': {
            'log_file': 'Hadoop/Hadoop_2k.log',
            'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        },

        'Spark': {
            'log_file': 'Spark/Spark_2k.log',
            'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        },

        'Zookeeper': {
            'log_file': 'Zookeeper/Zookeeper_2k.log',
            'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        },
        'OpenStack': {
            'log_file': 'OpenStack/OpenStack_2k.log',
            'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        },

        'BGL': {
            'log_file': 'BGL/BGL_2k.log',
            'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        },

        'HPC': {
            'log_file': 'HPC/HPC_2k.log',
            'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        },

        'Thunderbird': {
            'log_file': 'Thunderbird/Thunderbird_2k.log',
            'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        },

        'Windows': {
            'log_file': 'Windows/Windows_2k.log',
            'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        },

        'Linux': {
            'log_file': 'Linux/Linux_2k.log',
            'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        },

        'Mac': {
            'log_file': 'Mac/Mac_2k.log',
            'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        },

        'Android': {
            'log_file': 'Android/Android_2k.log',
            'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        },

        'HealthApp': {
            'log_file': 'HealthApp/HealthApp_2k.log',
            'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        },

        'Apache': {
            'log_file': 'Apache/Apache_2k.log',
            'log_format': '\[<Time>\] \[<Level>\] <Content>',
        },

        'Proxifier': {
            'log_file': 'Proxifier/Proxifier_2k.log',
            'log_format': '\[<Time>\] <Program> - <Content>',
        },

        'OpenSSH': {
            'log_file': 'OpenSSH/OpenSSH_2k.log',
            'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        },
    }
    Partition1 = ['Hadoop', 'HDFS', 'HealthApp', 'Mac', 'OpenSSH', 'Proxifier', 'Thunderbird', 'Windows']
    Partition2 = ['Android', 'Apache', 'BGL', 'HPC', 'Linux', 'OpenStack', 'Spark', 'Zookeeper']

    dataset_ = []
    for dataset in benchmark_settings.keys():
    # for dataset in Partition1:
        print(dataset)
        if (dataset_ and dataset not in dataset_):
            continue
        setting = benchmark_settings[dataset]

        input_dir = r'D:\log-parsing\Hooglle\logs\{}'.format(dataset)
        # output_dir = r'D:\log-parsing\Hooglle\result1'
        output_dir = r'D:\log-parsing\Hooglle\result2'
        log_file = '{}_2k.log'.format(dataset)

        log_format = setting['log_format']

        parser = LogParser(indir=input_dir, outdir=output_dir, log_format=log_format)

        label = parser.parse(log_file)
