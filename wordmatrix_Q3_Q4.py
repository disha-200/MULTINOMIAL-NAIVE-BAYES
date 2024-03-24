import math
from utils import *


def pr_wk_yi_word(freq_dict, totalwords_class, instance, vocab, laplace_smoothing):
    total_pr_word = 0.0
    for word in instance:
        if word in freq_dict:
            word_count = float(freq_dict[word])
        else:
            word_count = 0.0
        # pr_word = word_count/totalwords_instance
        pr_word = math.log((word_count + laplace_smoothing) / (float(totalwords_class) +
                                                               (laplace_smoothing * float(len(vocab)))))
        # print(pr_word)
        total_pr_word += pr_word
    # print(math.exp(total_pr_word), "total pr word")
    return total_pr_word


def pr_wk_yi_word_nolog(freq_dict, totalwords_class, instance, vocab, laplace_smoothing):
    total_pr_word = 1.0
    for word in instance:
        if word in freq_dict:
            word_count = float(freq_dict[word])
        else:
            word_count = 0.0
        # pr_word = word_count/totalwords_instance
        pr_word = (word_count + laplace_smoothing) / (float(totalwords_class) + (laplace_smoothing * float(len(vocab))))
        # print(pr_word)
        total_pr_word *= pr_word
    # print(math.exp(total_pr_word), "total pr word")
    return total_pr_word


# probability of class of each pos/neg train/test instance
def pr_yi(len_instance, instance, other_instance):
    total = float(len(instance)) + float(len(other_instance))
    class_count = float(len_instance)
    pr_class = class_count / total
    return pr_class


# creating dictionary for each instance and storing words and their freqs
def frequencies(class_instance):
    freq = {}
    for inner_list in class_instance:
        for element in inner_list:
            if element in freq:
                freq[element] += 1
            else:
                freq[element] = 1
    return freq


# classify instance into positive or negative class
def evaluate_metrics(predicted_class, actual_class):
    truepositive = 0
    truenegative = 0
    falsepositive = 0
    falsenegative = 0

    for flag in range(len(predicted_class)):
        if predicted_class[flag] == 1 and actual_class[flag] == 1:
            truepositive += 1
        elif predicted_class[flag] == 0 and actual_class[flag] == 0:
            truenegative += 1
        elif predicted_class[flag] == 1 and actual_class[flag] == 0:
            falsepositive += 1
        else:
            falsenegative += 1

    # print(len(predicted_class))
    accuracy = (truepositive + truenegative) / len(predicted_class)
    precision = truepositive / (truepositive + falsepositive)
    recall = truepositive / (truepositive + falsenegative)
    confusion_matrix = [[truepositive, falsenegative], [falsepositive, truenegative]]

    print("Accuracy", accuracy)
    print("Precision", precision)
    print("Recall", recall)
    print("Confusion Matrix")
    print(confusion_matrix)


def train(message, ls, pos_test, neg_test, pr_yi_pos_train, pr_yi_neg_train, freq_pos_train, freq_neg_train,
          totalwords_pos_train, totalwords_neg_train, vocab):
    actual_class = []
    predicted_class = []
    for instance in pos_test:
        pos_pr = math.log(pr_yi_pos_train) + pr_wk_yi_word(freq_pos_train, totalwords_pos_train, instance, vocab, ls)
        neg_pr = math.log(pr_yi_neg_train) + pr_wk_yi_word(freq_neg_train, totalwords_neg_train, instance, vocab, ls)
        if pos_pr > neg_pr:
            predicted_class.append(1)
        else:
            predicted_class.append(0)
        actual_class.append(1)

    for instance in neg_test:
        pos_pr = math.log(pr_yi_pos_train) + pr_wk_yi_word(freq_pos_train, totalwords_pos_train, instance, vocab, ls)
        neg_pr = math.log(pr_yi_neg_train) + pr_wk_yi_word(freq_neg_train, totalwords_neg_train, instance, vocab, ls)
        if pos_pr > neg_pr:
            predicted_class.append(1)
        else:
            predicted_class.append(0)
        actual_class.append(0)

    print(message, "\n")
    # print(len(predicted_class), predicted_class.count(1), predicted_class.count(0))
    # print(len(actual_class), actual_class.count(1), actual_class.count(0))
    evaluate_metrics(predicted_class, actual_class)


def naive_bayes():
    percentage_positive_instances_train = 1.0000
    percentage_negative_instances_train = 1.0000

    percentage_positive_instances_test = 1.0000
    percentage_negative_instances_test = 1.0000

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train,
                                                      percentage_negative_instances_train)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    with open('vocab.txt', 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)
    print("Vocabulary (training set):", len(vocab))

    # calculating the frequency of words and total words
    # in all the positive and  negative test train instances
    freq_pos_train = frequencies(pos_train)
    totalwords_pos_train = sum(freq_pos_train.values())

    freq_neg_train = frequencies(neg_train)
    totalwords_neg_train = sum(freq_neg_train.values())

    # probability of the class
    pr_yi_pos_train = pr_yi(len(pos_train), pos_train, neg_train)
    pr_yi_neg_train = pr_yi(len(neg_train), neg_train, pos_train)

    ls = 10
    message = f"\nWith highest alpha = {ls} and 100% training data"
    train(message, ls, pos_test, neg_test, pr_yi_pos_train, pr_yi_neg_train, freq_pos_train,
          freq_neg_train, totalwords_pos_train, totalwords_neg_train, vocab)

    percentage_positive_instances_train_4 = 0.5000
    percentage_negative_instances_train_4 = 0.5000

    percentage_positive_instances_test = 1.0000
    percentage_negative_instances_test = 1.0000

    (pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train_4,
                                                      percentage_negative_instances_train_4)
    (pos_test, neg_test) = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

    print('\n')
    print("Number of positive training instances:", len(pos_train))
    print("Number of negative training instances:", len(neg_train))
    print("Number of positive test instances:", len(pos_test))
    print("Number of negative test instances:", len(neg_test))

    with open('vocab.txt', 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)
    print("Vocabulary (training set):", len(vocab))

    # calculating the frequency of words and total words
    # in all the positive and  negative test train instances
    freq_pos_train = frequencies(pos_train)
    totalwords_pos_train = sum(freq_pos_train.values())

    freq_neg_train = frequencies(neg_train)
    totalwords_neg_train = sum(freq_neg_train.values())

    # probability of the class
    pr_yi_pos_train = pr_yi(len(pos_train), pos_train, neg_train)
    pr_yi_neg_train = pr_yi(len(neg_train), neg_train, pos_train)

    ls = 10
    message = f"\nWith highest alpha = {ls} and 50% training data"
    train(message, ls, pos_test, neg_test, pr_yi_pos_train, pr_yi_neg_train, freq_pos_train,
          freq_neg_train, totalwords_pos_train, totalwords_neg_train, vocab)


if __name__ == "__main__":
    naive_bayes()
