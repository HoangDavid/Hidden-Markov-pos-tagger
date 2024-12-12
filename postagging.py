import numpy as np
import time
import re

'''
Note to self:
Forward Algo/ Backward Algo -> this is to calculate the likelihood of a sequence of an observation

Viterbi Algo -> this is to find the best sequence of hidden states given the sequence of O
'''
def process_file(filename):
    tag_count = {}
    word_count = {}
    transition_count = {}
    emission_count = {}

    # Tag_counter for the first word of a sentence
    f_tag_counter = {}
    f_tag_total = 0

    with open (filename, 'r') as file:
        prev = None
        first = None
        for line in file:
            tokens = line.strip().split()
            if len(tokens) == 2:
                tokens[0] = tokens[0].lower()
                if first == None:
                    first = tokens[1]

                # Get all the unique words
                if tokens[0] not in word_count:
                    word_count[tokens[0]] = 1
                else:
                    word_count[tokens[0]] += 1

                # Count the number of each tag
                if tokens[1] not in tag_count:
                    tag_count[tokens[1]] = 1
                else:
                    tag_count[tokens[1]] += 1   

                # Count the emissions
                if tokens[1] not in emission_count:
                    emission_count[tokens[1]] = {}
                                        
                if tokens[0] not in emission_count[tokens[1]]:
                    emission_count[tokens[1]][tokens[0]] = 1
                else:
                    emission_count[tokens[1]][tokens[0]] += 1

                # Count the transitions
                if prev != None:
                    if tokens[1] not in transition_count[prev]:
                        transition_count[prev][tokens[1]] = 1
                    else:
                        transition_count[prev][tokens[1]] += 1
                    
                    prev = tokens[1]
                    if prev not in transition_count:
                        transition_count[prev] = {}
                else:
                    prev = tokens[1]
                    transition_count[prev] = {}

                # first word tag counter:
                if tokens[0] in ['.', '?', '!']:
                    if first not in f_tag_counter: 
                        f_tag_counter[first] = 1
                    else:
                        f_tag_counter[first] += 1
                    
                    first = None
                    f_tag_total += 1



    return transition_count, emission_count, tag_count, word_count, f_tag_counter, f_tag_total



def transition_matrix(transition_count, tags):
    size = len(tags)
    tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
    matrix = np.zeros((size, size))

    # normalization
    for prev_tag in transition_count:
        row_sum = sum(transition_count[prev_tag].values())
        for curr_tag in transition_count[prev_tag]:
            matrix[tag_to_idx[prev_tag]][tag_to_idx[curr_tag]] = transition_count[prev_tag][curr_tag] / row_sum

    return matrix, tag_to_idx



def emission_matrix(emission_count, tag_count, word_count):
    height = len(tag_count)
    width = len(word_count)
    
    tag_to_idx = {tag:idx for idx, tag in enumerate(tag_count)}
    word_to_idx = {word:idx for idx, word in enumerate(word_count)}

    matrix = np.zeros((height, width))

    # normalization
    for tag in emission_count:
        row_sum = sum(emission_count[tag].values())
        for word in emission_count[tag]:
            matrix[tag_to_idx[tag]][word_to_idx[word]] = emission_count[tag][word] / row_sum
    return matrix, word_to_idx

def init(tag_count, f_tag_count, f_tag_total):    
    # Initialization
    init_prob = {}

    for tag in tag_count:
        if tag in f_tag_count: 
            init_prob[tag] = f_tag_count[tag] / f_tag_total
        else:
            init_prob[tag] = 0

    return init_prob


def viterbi(trans_mx, emiss_mx, tag_count, tag_to_idx, word_to_idx, init_prob, obs):
    # Forward: the most probable path for each start of tag
    smoothing_factor = 1e-6 # To handle OOV case (out of vocab)

    states = tag_count
    prob = {}
    prev = {}
    for t in range(len(obs)):
        prob[t] = {}
        prev[t] = {}
        for state in states:
            prob[t][state] = 0
            prev[t][state] = 0

    for state in states:
        if obs[0] in word_to_idx:
            prob[0][state] = init_prob[state] * emiss_mx[tag_to_idx[state]][word_to_idx[obs[0]]]
        else:
            prob[0][state] = init_prob[state] * smoothing_factor
    
    for t in range(1,len(obs)):
        for s in states:
            for r in states:
                if obs[t] in word_to_idx:
                    emission_prob = emiss_mx[tag_to_idx[s]][word_to_idx[obs[t]]]
                else:
                    emission_prob = smoothing_factor

                new_prob = prob[t - 1][r] * trans_mx[tag_to_idx[r]][tag_to_idx[s]] * emission_prob
                
                if new_prob >= prob[t][s]:
                    prob[t][s] = new_prob
                    prev[t][s] = r

    # Backtracking
    path = [''] * len(obs)
    max_prob = float('-inf')

    tmp = None
    for state in states:
        if max_prob < prob[len(obs) - 1][state]:
            max_prob = prob[len(obs) - 1][state]
            path[len(obs) - 1] = state

    for t in range(len(obs) - 2, -1, -1):
        if path[t + 1] == '':
            print(tmp)
        path[t] = prev[t+1][path[t + 1]]

    return path

def test(filename, trans_mx, emiss_mx, tag_count, tag_to_idx, word_to_idx, init_prob,):

    sentences = {0:[]}
    counter = 0
    answer = {0:[]}
    with open(filename) as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) == 2:
                tokens[0] = tokens[0].lower() 
                if tokens[0] in ['.', '?', '!']:
                    sentences[counter].append(tokens[0])
                    answer[counter].append(tokens[1])
                    counter += 1
                    sentences[counter] = []

                    answer[counter] = []
                else:
                    sentences[counter].append(tokens[0])
                    answer[counter].append(tokens[1])


    c = 0
    correct = 0
    incorrect = 0
    for i in sentences:
        if sentences[i] != []:
            pred = viterbi(trans_mx, emiss_mx, tag_count, tag_to_idx, word_to_idx, init_prob, sentences[i])
            for k in range(len(pred)):
                c += 1
                if pred[k] == answer[i][k]:
                    correct += 1
                else:
                    incorrect += 1

    o = correct / c

    return f'Accuracy:{o}; # Correct: {correct}; # Incorrect:{incorrect}; Total: {c}'

def main():
    # start the timer
    start_time = time.time()

    # Training the model
    transition_count, emission_count, tag_count, word_count, f_tag_counter, f_tag_total = process_file('WSJ_02-21.pos')
    trans_mx, tag_to_idx = transition_matrix(transition_count, tag_count)
    emiss_mx, word_to_idx = emission_matrix(emission_count, tag_count, word_count)

    # Testing the model
    init_prob = init(tag_count, f_tag_counter, f_tag_total)
    test_sentence = input("sample:")
    obs = re.findall(r'\w+|[^\w\s]', test_sentence)

    path = viterbi(trans_mx, emiss_mx, tag_count, tag_to_idx, word_to_idx, init_prob, obs)
    print(f'output: {path}')
    print(test('WSJ_24.pos',trans_mx, emiss_mx, tag_count, tag_to_idx, word_to_idx, init_prob))

    print()
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
        

