import json
import os
import math


def byuccl(filepath, writefile):
    # Load JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)
    # filepath_arr = filepath.split('/')
    # write_file=""
    # for i in range(len(filepath_arr)-1):
    #     write_file+=filepath_arr[i]+"/"
    # filename = filepath_arr[len(filepath_arr)-1].split('.')[0]+".txt"
    # write_file+=filename
    # Modify data
    for cid in data:
        messages = data[cid]["messages"]
        ln = len(messages)
        if(ln % 2 != 0):
            ln -= 1
        for i in range(ln):
            each = messages[i]
            string = ""
            counter = 0
            for idv in each:
                string += idv["text"]+" "
                counter += 1
                if(counter > 3):
                    break
            if(i % 2 == 0):
                string = "User: "+string
                with open(writefile, 'a') as f:
                    f.write(string.strip()+'\n')
            else:
                string = "Bot: "+string
                with open(writefile, 'a') as f:
                    f.write(string.strip()+'\n')
                    f.write('\n')


def split_in_sets(filepath, writepath, train_ratio, test_ratio):
    with open(filepath, 'r') as file:
        # Read all the lines into a list
        lines = file.readlines()

    total_lines = len(lines)
    train_lines = (0, math.floor(total_lines * train_ratio))
    test_lines = (train_lines[1], math.floor(
        total_lines * (test_ratio+train_ratio)))
    valid_lines = (test_lines[1], math.floor(total_lines))

    train_path = writepath+'_train.txt'
    test_path = writepath+'_test.txt'
    valid_path = writepath+'_valid.txt'

    with open(train_path, 'w') as train_file:
        data = lines[train_lines[0]:train_lines[1]]
        train_file.writelines(data)

    with open(test_path, 'w') as test_file:
        data = lines[test_lines[0]:test_lines[1]]
        test_file.writelines(data)

    with open(valid_path, 'w') as valid_file:
        data = lines[valid_lines[0]:valid_lines[1]]
        valid_file.writelines(data)


def checkData(filepath):
    import numpy as np
    data = np.loadtxt(filepath)
    if np.isnan(data).any() or np.isinf(data).any():
        print('The input data contains NaN or Inf values')
    else:
        print('The input data is valid')


def getVocabSize(filepath):
    with open(filepath, 'r') as f:
        data = f.read()
    all_words = data.split()
    size = len(all_words)
    vocab = []
    for word in all_words:
        if(word not in vocab):
            vocab.append(word)
    return len(vocab)


def getMaxSequenceSize(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    max1 = 0
    max2 = 0
    line1 = ""
    line2 = ""
    for line in data:
        length = len(line)
        if(length > max1):
            max1 = length
            line1 = line
        elif(length > max2):
            max2 = length
            line2 = line
    return max1+max2


def cleanup(readpath, writepath):
    with open(readpath, 'r') as file:
        lines = file.readlines()

    with open(writepath, 'w') as file:
        for line in lines:
            if not line.startswith('hit:'):
                file.write(line)


def prepare_test_data(readpath):
    with open(readpath, 'r') as f:
        test_data = f.readlines()
    input = []
    response = []
    input_text = ""
    response_text = ""
    for data in test_data:
        if(data.startswith("User:")):
            response.append(response_text)
            respose_text = ""
            input_text += data.strip()
        elif(data.startswith("Bot:")):
            input.append(input_text)
            input_text = ""
            response_text += data.strip()
    response.pop(0)
    return input, response


def prepare(readpath):
    with open(readpath, 'r') as f:
        test_data = f.readlines()
    input = {}
    response = {}
    conv_id = 1
    idx = 1
    for i in range(len(test_data)):
        input[idx] = ""
        response[idx] = ""
        if(test_data[i] == "\n" or test_data[i] == '\n'):
            idx += 1
    for data in test_data:
        if(conv_id >= idx):
            break
        if(data.startswith("User")):
            input[conv_id] += data
        elif(data.startswith("Bot")):
            response[conv_id] += data
        else:
            conv_id += 1
    return input, response


prepare("datasets/byupccl/combined_test.txt")
# prepare_test_data("datasets/byupccl/combined_test.txt")
# checkData("datasets/byupccl/byupccl_train.txt")
# split_in_sets("datasets/byupccl_extra/byupccl.txt","datasets/byupccl/byupccl", 0.8, 0.1)

# byuccl("datasets/byupccl_extra/dataset.json", "datasets/byupccl_extra/byupccl.txt")

# print(getVocabSize("datasets/byupccl/byupccl_train.txt"))
# print(getMaxSequenceSize("datasets/byupccl/byupccl_train.txt"))

# cleanup("datasets/byupccl/byupccl_train.txt", "datasets/byupccl/byupccl_train.txt")
