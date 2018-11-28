# -*- coding: utf-8 -*-
import sys, codecs

SRC_FILE = sys.argv[1]
CONTENT_PLAN = sys.argv[2]
EVAL_OUTPUT = sys.argv[3]
CONTENT_PLAN_INTER = sys.argv[4]

TRAIN = True
DELIM = u"￨"

inputs = []
content_plans = []
with codecs.open(CONTENT_PLAN, "r", "utf-8") as corpus_file:
    for _, line in enumerate(corpus_file):
        content_plans.append(line.split())
with codecs.open(SRC_FILE, "r", "utf-8") as corpus_file:
    for _, line in enumerate(corpus_file):
        inputs.append(line.split())

outputs = []
eval_outputs = []
for i, input in enumerate(inputs):
    content_plan = content_plans[i]
    output = []
    eval_output = []
    records = set()
    for record in content_plan:
        output.append(input[int(record)].encode("utf-8"))
        elements = input[int(record)].split(DELIM)
        if elements[0].isdigit():
            record_type = elements[2]
            if not elements[2].startswith('TEAM'):
                record_type = 'PLAYER-'+ record_type
            eval_output.append("|".join([elements[1].replace("_"," "), elements[0], record_type]))
    outputs.append(" ".join(output))
    eval_outputs.append("\n".join(eval_output))

output_file = open(CONTENT_PLAN_INTER, 'w')
output_file.write("\n".join(outputs))
output_file.write("\n")
output_file.close()

output_file = open(EVAL_OUTPUT, 'w')
output_file.write("\n")
output_file.write("\n\n".join(eval_outputs))
output_file.write("\n")
output_file.close()
