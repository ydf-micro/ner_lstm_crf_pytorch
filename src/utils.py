# *_*coding:utf-8 *_*

def f1_score(true_path, predict_path, lengths):
    batch_TP_FP = 0
    batch_TP_FN = 0 # not division by zero
    batch_TP = 0
    for true, predict, len in zip(true_path, predict_path, lengths):
        true = true[:len]  # remove the 0 padding

        TP_FP = 0
        TP_FN = 0
        TP = 0
        for i in predict:
            if i == 3 or i == 5 or i == 7:
                TP_FP += 1

        for i in true:
            if i == 3 or i == 5 or i == 7:
                TP_FN += 1

        for i, index in enumerate(true):
            if predict[i] == index and index != 2:
                if index == 3 or index == 5 or index == 7:
                    TP += 1
        # print('------')
        # print(TP)
        # print(TP_FP)
        # print(TP_FN)
        # print(true)

        batch_TP_FP += TP_FP
        batch_TP_FN += TP_FN
        batch_TP += TP

    precision = batch_TP / batch_TP_FP
    recall = batch_TP / batch_TP_FN
    f1 = 2 * precision * recall / (precision + recall + 1)

    print(f'precision: {precision:.2f}, '
          f'recall: {recall:.2f}, f1:{f1:.2f}')
