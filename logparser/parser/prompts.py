CONTENT_INQUIRY="""You are an operations and maintenance expert, and now we need you to analyze the log.
I will provide a log and some of its content. Please analyze what the content "{content}" means in the log "{log}".
Please only analyze the given content and do not conduct additional analysis.
answer:
"""


CONSTANT_VARIABLE_INQUIRY="""You are an operations and maintenance expert. Now determine whether the given content is a constant or a variable based on the following principles.
1. If a content constitutes a logging framework, it is a constant.
2. If a content is the value filled in at runtime (e.g. URL, instance name, uesr name, ip, path, id), then it is a variable.
3. If a content is the value varies with time (timestamp, date), then it is a variable.
4. If a content is enumerable (e.g. state, type, etc.) then it is a constant.
Note that if a content has a fixed meaning, but there are infinite other contents to choose from (e.g. user name root), then it is a variable.
Known that: {analyse}
Please analyze what the content "{content}" in the log "{log}" is a "constant" or a "variable":
Please only reply "constant" or a "variable" in answer.
answer:
"""



MERGE_TWO_INQUIRY="""You are an operations and maintenance expert. Now we have two preliminarily parsed templates from logs, and we need you to determine whether they belong to the same template based on the following principles.
1. If the difference between the two templates is the different values of the same runtime variable, then they are the same template and you have to reply yes.
2. If the difference between the two templates is logging framework, then they are the different template and you have to reply no.
3. If the difference between the two templates is supplementary explanation and there is no additional new information (e.g. conversion of Units), then they are the same template and you have to reply yes.
4. If the difference between the two templates is property name, then they are the different template and you have to reply no.
5. If the difference between the two templates is enumerable (e.g. state, type, etc.) then they are the different template and you have to reply no.

Please analyze whether the following two preliminarily templates belong to the same template:
{template1}
{template2}
Please reply your analyse in thought and only reply "yes" or a "no" in answer.
thought:
answer:
"""


