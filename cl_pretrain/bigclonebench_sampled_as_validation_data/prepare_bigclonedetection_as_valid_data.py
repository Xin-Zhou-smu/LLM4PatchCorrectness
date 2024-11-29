from datasets import load_dataset
from collections import Counter
import random

"""
As we lack semantically equivalent code changes, we sample 1,000 pairs of code from BigCodeBench. Each pair represents a semantically equivalent code snippet.

For each code snippet, we randomly delete some lines and then add them back to create a code change. 

If the two code changes originate from a pair of semantically equivalent code,  these two code change pairs are considered similar, and we assign them a similarity score of 3.

If two code changes originate from the same function, we assume these changes are closely related and assign them a similarity score of 5.
"""


# Load the dataset from Hugging Face
dataset = load_dataset('nchen909/bigclonebench-processed')
train_dataset = dataset['test']
clone_pair = []
clone_labels = []

positive_count, negative_count = 0, 0
for i in range(50000):
    label = train_dataset[i]['label']
    func1 = train_dataset[i]['func1']
    func2 = train_dataset[i]['func2']


    if func1+'SpecialSeperatorForCodePair'+func2 not in clone_pair:
        if (label == 1 and positive_count<500) or (label == 0 and negative_count<500):
            clone_pair.append(func1+'SpecialSeperatorForCodePair'+func2)
            clone_labels.append(label)
            if label == 1:
                positive_count +=1
            elif label == 0:
                negative_count +=1

    if positive_count>=500 and negative_count>=500:
        break


print(Counter(clone_labels))
print(len(clone_labels))
print(len(clone_pair))

def randomly_add_deleted_added_lines(code):
    lines = code.split('\n')
    random_number = random.randint(0, len(lines)-1)
    new_lines = lines[0:random_number] + ['-  '+lines[random_number]] + ['+  '+lines[random_number]] + lines[random_number+1:]
    new_code = '\n'.join(new_lines)

    return new_code




all_validation_lines = []
for i in range(len(clone_labels)):
    data_index = 'bigclonebench-test-'+str(i)
    func1 = clone_pair[i].split('SpecialSeperatorForCodePair')[0].replace(';', ';\n').replace('{', '{\n').replace('}','}\n')
    func2 = clone_pair[i].split('SpecialSeperatorForCodePair')[1].replace(';', ';\n').replace('{', '{\n').replace('}','}\n')
    label = clone_labels[i]

    revised_fun1 = randomly_add_deleted_added_lines(func1)
    revised_fun1_ = randomly_add_deleted_added_lines(func1)
    if label == 1:
        line = data_index + '|SpecialSeperatorForDevSet|' + randomly_add_deleted_added_lines(func1).replace('\n','\\n') + '|SpecialSeperatorForDevSet|' + randomly_add_deleted_added_lines(func1).replace('\n', '\\n') + '|SpecialSeperatorForDevSet|' + '5'
        all_validation_lines.append(line)
        line = data_index + '|SpecialSeperatorForDevSet|' + randomly_add_deleted_added_lines(func1).replace('\n','\\n') + '|SpecialSeperatorForDevSet|' + randomly_add_deleted_added_lines(func2).replace('\n', '\\n') + '|SpecialSeperatorForDevSet|' + '3'
        all_validation_lines.append(line)
    else:
        line = data_index + '|SpecialSeperatorForDevSet|' + randomly_add_deleted_added_lines(func1).replace('\n','\\n') + '|SpecialSeperatorForDevSet|' + randomly_add_deleted_added_lines(func2).replace('\n', '\\n') + '|SpecialSeperatorForDevSet|' + '0'
        all_validation_lines.append(line)


print(len(all_validation_lines))
print (Counter([l.split('|SpecialSeperatorForDevSet|')[-1] for l in all_validation_lines]))

with open('bigclonebench_sampled_as_validation_data/valid.txt', 'w') as f:
    for l in all_validation_lines:
        f.write(l.strip()+'\n')