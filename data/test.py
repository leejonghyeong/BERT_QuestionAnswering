def find_answer_indices(answer, context, sep_token_index):
    for i in range(len(context)):
        if answer == context[i: i + len(answer)]:
            return [i + sep_token_index, i + sep_token_index + len(answer) - 1]

import json

infile = "./SQuAD_train_example2.json"
outfile = "./SQuAD_no_answer"

cnt = 0
'''
with open(infile, "rt") as f:
    with open(outfile, "wt") as g:
        for line in f:
            instance = json.loads(line)

            answer_indices= instance["answer_indices"]

            if answer_indices == None:
                cnt += 1
                g.write(line + "\n")
'''

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
'''
outfile2 = "./SQuAD_no_answer_text.txt"
with open(outfile, "rt") as f:
    with open(outfile2, "wt") as g:
        for line in f:
            instance = json.loads(line)
            q = instance["question"]
            a = instance['answer']
            c = instance["context"]

            if q != None:
                q = tokenizer.decode(q)
            if a != None:
                a = tokenizer.decode(a)
            if c != None:
                c = tokenizer.decode(c)

            g.write("question: "+q+"\n")
            g.write("answer: "+a+"\n")
            g.write("context: "+c+"\n")
'''

A = {"answer": [2416], "context": [2012, 1996, 26898, 3296, 8922, 2982, 1010, 20773, 2363, 2702, 9930, 1010, 2164, 2201, 1997, 1996, 2095, 2005, 1045, 2572, 1012, 1012, 1012, 14673, 9205, 1010, 2501, 1997, 1996, 2095, 2005, 1000, 17201, 1000, 1010, 1998, 2299, 1997, 1996, 2095, 2005, 1000, 2309, 6456, 1006, 2404, 1037, 3614, 2006, 2009, 1007, 1000, 1010, 2426, 2500, 1012, 2016, 5079, 2007, 21360, 18143, 2940, 2005, 2087, 8922, 9930, 1999, 1037, 2309, 2095, 2011, 1037, 2931, 3063, 1012, 1999, 2230, 1010, 20773, 2001, 2956, 2006, 3203, 23332, 1005, 1055, 2309, 1000, 7026, 1000, 1998, 2049, 2189, 2678, 1012, 1996, 2299, 9370, 1996, 2149, 3769, 2774, 3673, 1010, 3352, 1996, 4369, 2193, 1011, 2028, 2005, 2119, 20773, 1998, 23332, 1010, 15233, 2068, 2007, 3814, 2232, 11782, 2005, 2087, 2193, 1011, 3924, 2144, 1996, 13188, 2327, 2871, 15341, 3673, 3390, 1999, 2826, 1012, 1000, 7026, 1000, 2363, 1037, 8922, 2400, 6488, 2005, 2190, 3769, 5792, 2007, 2955, 1012], "question": [2129, 2116, 2193, 2028, 3895, 2106, 20773, 2085, 2031, 2044, 1996, 2299, 1000, 7026, 1000, 1029], "answer_indices": None}

print(tokenizer.decode(A["answer"]))
print(tokenizer.decode(A["context"]))
print(tokenizer.decode(A["question"]))
print(tokenizer.decode([4369 ]))
