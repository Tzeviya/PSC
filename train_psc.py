import torch.nn.functional as F
import torch
import numpy as np
import math
import time
import pdb

NUM_CLASSES = 1000

def pool_and_loss(y, output, r, is_cuda ):

    t_cuda= torch.cuda if is_cuda else torch

    #log-sum-exp
    T = output.size(3)

    log_sum_exp = (1.0 / r) * ( torch.log(t_cuda.FloatTensor([1.0/T])) + torch.logsumexp( r * output, dim = 3 , keepdim = False))
    log_sum_exp = torch.squeeze(log_sum_exp)
    #sig_layer = torch.sigmoid(log_sum_exp)
    sig_layer = log_sum_exp
    #print('sig_layer {}'.format(sig_layer))
    #loss
    mult = torch.mul(y, sig_layer)
    loss = torch.log(1 + torch.exp(-mult)) #same size as mult (batchsize x classes )
    loss = torch.sum(loss, dim = 1) #? TODO: check dim
    loss = torch.mean(loss)

    return loss

def train(train_loader, model, r, optimizer, epoch, is_cuda, log_interval, print_progress=True):

    t_cuda= torch.cuda if is_cuda else torch
    model.train()
    global_epoch_loss = 0
    for batch_idx, (data, target, location, idx) in enumerate(train_loader):
        #print('starting batch')
        #print(time.time())
        #pdb.set_trace()
        if is_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = pool_and_loss(target, output, r, is_cuda)
        
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.item()
        #print('ending batch')
        #print(time.time())
        if print_progress:
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100.
                    * batch_idx / len(train_loader), loss.item()))
    return global_epoch_loss / len(train_loader.dataset)


def test(loader, model, r, is_cuda):
    t_cuda= torch.cuda if is_cuda else torch
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for batch_idx, (data, target, location, idx) in enumerate(loader):
        
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = pool_and_loss(target, output, r, is_cuda)
            total_loss += loss

    print('Val Loss: {}'.format(total_loss))
    return total_loss


def test_acc(loader, model, wav_len, r, threshold, is_cuda):
    t_cuda= torch.cuda if is_cuda else torch
    with torch.no_grad():
        model.eval()
        total_counter_oracle = 0.0
        total_correct_oracle = 0.0
        total_actual_acc = 0.0
        total_f1 = 0.0
        total_acc_per_term = np.zeros((NUM_CLASSES,3)) #np.zeros((NUM_CLASSES,3)) #tp, fp, fn
        total_jointly_acc = np.zeros(2)
        for batch_idx, (data, target, location, idx) in enumerate(loader):
            
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            
            #pdb.set_trace()
            count_corrects_oracle, counter_oracle = eval_oracle(output,target, location, wav_len, r, threshold = threshold)#Val Acc: 0.925275510236 on 1000 examples
            #actual_batch_acc, f1_score, acc_per_term = eval_actual(output, target, location, r, is_cuda, threshold = threshold)
            acc_per_term, jointly_acc_per_batch = eval_actual(output, target, location, wav_len, r, is_cuda, threshold = threshold) #batch_acc: for article (recall?)
            #total_acc += batch_acc_oracle
            total_correct_oracle += count_corrects_oracle
            total_counter_oracle += counter_oracle
            #total_actual_acc += actual_batch_acc
            #total_f1 += f1_score
            total_acc_per_term += acc_per_term
            total_jointly_acc += jointly_acc_per_batch
            #pdb.set_trace()

    f1_per_term = np.zeros(NUM_CLASSES)
    #calculate F1 score for each class
    for item in range(len(total_acc_per_term)):
        if total_acc_per_term[item][0] + total_acc_per_term[item][1] + total_acc_per_term[item][2] == 0:
                continue #zeros
        f1_per_term[item] = (2*total_acc_per_term[item][0]) / (2*total_acc_per_term[item][0] + total_acc_per_term[item][1] + total_acc_per_term[item][2])

    #precision: TP/(TP + FP)    
    temp_acc_sum = np.sum(total_acc_per_term, 0) #collapsing C dimension
    precision = float(temp_acc_sum[0]) / float(temp_acc_sum[0] + temp_acc_sum[1])
    recall = float(temp_acc_sum[0]) / float(temp_acc_sum[0] + temp_acc_sum[2])
    f1 = (2 * float(temp_acc_sum[0]))/ ( 2 * float(temp_acc_sum[0]) + float(temp_acc_sum[1]) + float(temp_acc_sum[2]))


    #print('Val Acc: {}'.format(float(total_acc)/len(loader)))
    print('threshold: {}'.format(threshold))
    print('Actual:')
    print('Val Acc: {}'.format(float(total_jointly_acc[0])/total_jointly_acc[1]))
    print('F1 mean: {}'.format(np.mean(f1)))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    
    print('Oracle:')
    print('Val Acc: {}'.format(float(total_correct_oracle)/total_counter_oracle))
    

    return float(total_jointly_acc[0])/total_jointly_acc[1] #val acc


def eval_actual(output, target, location, wav_len, r, is_cuda, threshold): #equation 5 of paper

    t_cuda= torch.cuda if is_cuda else torch
    sr = 16000
    time = wav_len #1.0 #one second

    #log-sum-exp
    T = output.size(3)
    log_sum_exp = (1.0 / r) * ( torch.log(t_cuda.FloatTensor([1.0/T])) + torch.logsumexp( r * output, dim = 3 , keepdim = False))
    log_sum_exp = torch.squeeze(log_sum_exp)

    #batch X num_of_classes
    sig_layer = torch.sigmoid(log_sum_exp) #P(w|x)

    if len(sig_layer.shape) == 1: #single batch
        sig_layer = sig_layer.unsqueeze(0)

    #elements greater than threshold
    mask = (torch.gt(sig_layer, threshold)).float()*sig_layer
    gt_thresh_idx = mask.nonzero()

    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    #F1_SCORE = (2TP)/(2TP + FP + FN) #wikipedia
    position_correct = 0.0
    counter = 0.0

    output = output.squeeze(2) #output: 32, 1000, 1, 96
    m_max, m_argmax = torch.max(output, dim=2)

    #for false negatives
    non_zero_indices = (target + 1).nonzero() #([[0, 0], [1, 1],...

    acc_per_term = np.zeros((NUM_CLASSES,3)) #tp, fp, fn
    f1_per_term = np.zeros(NUM_CLASSES)
    jointly_acc_per_batch= np.zeros(2)
    for line in gt_thresh_idx:
        batch_idx_pred = line[0]
        word_idx_pred = line[1] #class_idx

        #comparing with real label
        exists, start_frame, end_frame = location[batch_idx_pred, word_idx_pred]

        if exists == 1: #predicted that word exists, and it exists
            tp += 1
            acc_per_term[word_idx_pred][0] += 1

            #now check positiion
            frames_per_output_unit = float(sr * time) / output.size(2)
            start_unit = math.floor((start_frame/frames_per_output_unit).item())
            end_unit = math.ceil((end_frame/frames_per_output_unit).item()) #not including the end
            
            predict_frame = m_argmax[batch_idx_pred][word_idx_pred]

            if predict_frame >= start_unit and predict_frame < end_unit:
                position_correct += 1
            counter += 1
            

        else: #predicted that word exists, but it doesn't 
            fp += 1
            acc_per_term[word_idx_pred][1] += 1


    #iterating over uttered words
    fn = len(non_zero_indices) - tp

    count_existance = np.zeros(NUM_CLASSES)
    for batch_i, word_i in non_zero_indices:
        count_existance[word_i.item()] +=1
    #pdb.set_trace()
    for item in range(len(acc_per_term)):
        acc_per_term[item][2] = count_existance[item] - acc_per_term[item][0]

    #f1_score = (2*tp) / (2*tp + fp + fn)

    jointly_acc_per_batch[0], jointly_acc_per_batch[1] = position_correct, len(non_zero_indices) 
    #batch_acc = float(position_correct)/len(non_zero_indices) #true positive rate (TP/P)

    return acc_per_term, jointly_acc_per_batch


def eval_oracle(output, target, location, wav_len, r, threshold): #equation 6 of paper
    
    sr = 16000
    time = wav_len #1.0 #one second

    #network output: 32, 1000, 1, 96
    output = output.squeeze(2)
    m_max, m_argmax = torch.max(output, dim=2)

    count_corrects = 0
    counter = 0
    end_minus_start_avg = 0.0
    non_zero_indices = (target + 1).nonzero() #([[0, 0], [1, 1],...
    for line in non_zero_indices:
        batch_idx = line[0]
        word_idx = line[1] #class_idx
        exists, start_frame, end_frame = location[batch_idx, word_idx]

        frames_per_output_unit = float(sr * time) / output.size(2)
        start_unit = math.floor((start_frame/frames_per_output_unit).item())
        end_unit = math.ceil((end_frame/frames_per_output_unit).item()) #not including the end
        
        predict_frame = m_argmax[batch_idx][word_idx]

        if predict_frame >= start_unit and predict_frame < end_unit:
            count_corrects += 1
        counter += 1

        end_minus_start_avg += (end_unit - start_unit)/output.size(2)


    #batch_acc = float(count_corrects)/counter
    #end_minus_start_avg/=counter
    #print(end_minus_start_avg) - probability of 1/3 to be correct

    return count_corrects, counter


def keyword_spotting_test_mtwv(loader, model, wav_len, r, threshold, is_cuda):
    
    #pdb.set_trace()
    keyword_list =['any', 'easily', 'known', 'only', 'show', 'battle', 'fifty', 'land', 'perfect', 'thank', 'birds', 'filled', 'lie', 'perhaps', 'them', 'cannot', 'great', 'never', 'presence', 'years' ]
    keyword_indices = [loader.dataset.class_to_idx[keyword] for keyword in keyword_list]

    # print(zip(keyword_list, keyword_indices))

    t_cuda= torch.cuda if is_cuda else torch
    with torch.no_grad():
        model.eval()
        calcs_for_atwv_map = {}  #key: keyword. points at: list of 3: p_correct, p_spurious, N_true
        for batch_idx, (data, target, location, idx) in enumerate(loader):
            
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            #log-sum-exp
            T = output.size(3)
            log_sum_exp = (1.0 / r) * ( torch.log(t_cuda.FloatTensor([1.0/T])) + torch.logsumexp( r * output, dim = 3 , keepdim = False))
            log_sum_exp = torch.squeeze(log_sum_exp)

            #batch X num_of_classes
            sig_layer = torch.sigmoid(log_sum_exp) #P(w|x)

            if len(sig_layer.shape) == 1: #single batch
                sig_layer = sig_layer.unsqueeze(0)
            #pdb.set_trace()
            twv_variable_calculations(target, location, sig_layer, keyword_indices, calcs_for_atwv_map, threshold)

        t_speech = len(loader.dataset) * wav_len #total amount of speech in the test data (in seconds)
        this_atwv = mtwv(calcs_for_atwv_map, t_speech)

    print('ATWV: {}'.format(this_atwv)) 
    return this_atwv


def twv_variable_calculations(target, location, sig_layer, keyword_indices, calcs_for_atwv_map, threshold):

    #pdb.set_trace()
    predicted_scores_gt_thresh = (torch.gt(sig_layer, threshold)).float()*sig_layer
    gt_thresh_idx = predicted_scores_gt_thresh.nonzero()
    #for false negatives
    non_zero_indices = (target + 1).nonzero() #([[0, 0], [1, 1],...

    for batch, target_keyword in non_zero_indices:
        target_keyword = target_keyword.item()
        if target_keyword in keyword_indices:
            if not target_keyword in calcs_for_atwv_map:
                calcs_for_atwv_map[target_keyword] = [0, 0, 0]

            n_true = calcs_for_atwv_map[target_keyword][2]

            n_true+=1 #true number of occurences of term in corpus

            calcs_for_atwv_map[target_keyword][2] = n_true

    #pdb.set_trace()
    for batch_idx_pred, predict_keyword in gt_thresh_idx:
        predict_keyword = predict_keyword.item()
        if predict_keyword in keyword_indices:
            if not predict_keyword in calcs_for_atwv_map:
                calcs_for_atwv_map[predict_keyword] = [0, 0, 0]

            n_correct = calcs_for_atwv_map[predict_keyword][0]
            n_spurious = calcs_for_atwv_map[predict_keyword][1]
            
            exists, start_frame, end_frame = location[batch_idx_pred, predict_keyword]

            n_correct += exists.item()
            if exists.item() == 0: n_spurious += 1

            calcs_for_atwv_map[predict_keyword][0] = n_correct
            calcs_for_atwv_map[predict_keyword][1] = n_spurious



def mtwv(calcs_for_atwv_map, t_speech):


    c_over_v = 0.1 #from article
    pr_term = 0.0001
    beta = c_over_v * (math.pow(pr_term, -1.0) - 1);

    m_sum = 0.0
    keyword_not_in_dataset = 0.0
    for keyword, item in calcs_for_atwv_map.items():

        n_correct, n_spurious, n_true = item

        if n_true == 0:
            keyword_not_in_dataset += 1
            continue

        p_miss = 1.0 - (float(n_correct) / n_true)
        n_nnt = 1.0 * t_speech - n_true
        p_false_alarm = float(n_spurious) / n_nnt
        current_val = p_miss + beta * p_false_alarm

        m_sum += current_val
 

        #print
    #     print('========================print ATWV=========================')
    #     print('keyword: {}'.format(keyword))
    #     print('n_true: {}'.format(n_true))
    #     print('n_correct: {}'.format(n_correct))
    #     print('n_spurious: {}'.format(n_spurious))
    #     print('t_speech: {}'.format(t_speech))
    #     print('n_nnt: {}'.format(n_nnt))
    #     print('p_miss: {}'.format(p_miss))
    #     print('p_false_alarm: {}'.format(p_false_alarm))
    #     print('current atwv val: {}'.format(current_val))
    #     print('===========================================================')

    # print('kwd not in datset: {}'.format(keyword_not_in_dataset))
    average = m_sum/(len(calcs_for_atwv_map.keys()) - keyword_not_in_dataset) #divide by number of keywords
    return 1 - average



