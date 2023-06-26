import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from label_smooth import CE_Label_Smooth_Loss
from mmd import mmd
from cdd import cdd
from load_data import get_data_path, load4train
from network import Encoder, ClassClassifier

def set_seed(seed=10):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def test(test_loader, model, criterion, cuda):
    model.eval()
    correct = 0
    for _, (test_input, label) in enumerate(test_loader):
        if cuda:
            test_input, label = test_input.cuda(), label.cuda()
        test_input, label = Variable(test_input), Variable(label)
        output = model(test_input)
        loss = criterion(output, label)
        _, pred = torch.max(output, dim=1)
        correct += pred.eq(label.data.view_as(pred)).sum()
    accuracy = float(correct) / len(test_loader.dataset)
    
    return loss, accuracy

def get_src_center(source, src_label, num_class):
    sc = torch.empty(num_class, source.size(1)).cuda()
    for i in range(num_class):
        sc[i, :] = torch.mean(source[src_label==i], dim=0)
    return sc

def K_mean(target, tar_ct, num_class):
    batch_size = target.size(0)
    dist = torch.empty(batch_size, num_class).cuda()
    for cls in range(num_class):
        center = tar_ct[cls, :]
        center = center.repeat(batch_size,1)
        dist[:, cls] = 0.5*(1-torch.cosine_similarity(target, center, dim=1))
    dist_value, tar_label = torch.min(dist, dim=1) 

    for i in range(num_class):
        tar_ct[i, :] = torch.mean(target[tar_label==i], dim=0)
    return dist_value, tar_label, tar_ct


def main(data_loader_dict, net_config, optim_config, cuda, writer, seed=3):
    set_seed(seed)
    # Create the model
    encoder = Encoder(net_config["fts"])
    cls_classifier = ClassClassifier(net_config["cls"])

    # loss criterion
    # LOSS_WEIGHT = 0.3
    criterion = nn.CrossEntropyLoss()

    # Use GPU
    if cuda:
        encoder = encoder.cuda()
        cls_classifier = cls_classifier.cuda()
        criterion = criterion.cuda()

    # Optimizer
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(cls_classifier.parameters()),
        **optim_config
    )
    

    epochs = 600
    interval = 10
    best_acc = 0
    tar_center = None
    print("----------Starting training the model----------")
    # Begin training
    for epoch in range(1, epochs+1):
        encoder.train()
        cls_classifier.train()
        correct = 0
        count = 0

        if epoch % interval == 0:
            test_loss, acc = test(data_loader_dict["test_loader"], nn.Sequential(encoder, cls_classifier), criterion, cuda)
            if acc > best_acc:
                best_acc = acc
            print("Testing, Epoch: %d, accuracy: %f, best accuracy: %f" % (epoch, acc, best_acc))
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/Accuracy", acc, epoch)
            writer.add_scalar("test/Best_Acc", best_acc, epoch)
            
        for _, (src_examples, tar_examples) in enumerate(zip(data_loader_dict["source_loader"], data_loader_dict["target_loader"])):
            src_data, src_label_cls = src_examples
            tar_data, _ = tar_examples

            if cuda:
                src_data, src_label_cls = src_data.cuda(), src_label_cls.cuda()
                tar_data = tar_data.cuda()
            
            src_data, src_label_cls = Variable(src_data), Variable(src_label_cls)
            tar_data = Variable(tar_data)

            # encoder model forward
            src_feature = encoder(src_data)
            tar_feature = encoder(tar_data)

            ################################################
            # classification loss
            src_output_cls = cls_classifier(src_feature)
            tar_output_cls = cls_classifier(tar_feature)
            
            cls_loss = criterion(src_output_cls, src_label_cls)
            """
            if tar_center is None:
                tar_center = get_src_center(src_feature, src_label_cls, net_config["cls"])
            conf_val, pseudo_label, tar_center = K_mean(tar_feature, tar_center, net_config["cls"])

            confident_bool = conf_val <= 0.125
            confident_example = tar_feature[confident_bool]
            confident_label = pseudo_label[confident_bool]
            align_loss = cdd(src_feature, confident_example, src_label_cls, confident_label)
            """
            # align_loss = mmd(src_feature, tar_feature)
            # fake target label
            """
            tar_softmax_output = nn.functional.softmax(tar_output_cls, dim=1)
            max_prob, pseudo_label = torch.max(tar_softmax_output, dim=1)
            confident_bool = max_prob >= 0.75
            confident_example = tar_feature[confident_bool]
            confident_label = pseudo_label[confident_bool]
            
            align_loss = cdd(src_feature, confident_example, src_label_cls, confident_label)
            """
            # LOSS_WEIGHT = 1.0/(1.0+torch.exp(torch.tensor(100.0-epoch)))
            # # if epoch <= 100:
            # mmd_loss = mmd(src_feature, tar_feature)
            # elif epoch > 100:
            # fake target label
            # tar_softmax_output = nn.functional.softmax(tar_output_cls, dim=1)
            # max_prob, pseudo_label = torch.max(tar_softmax_output, dim=1)
            # confident_bool = max_prob >= 0.70
            # confident_example = tar_feature[confident_bool]
            # confident_label = pseudo_label[confident_bool]
            
            # cdd_loss = cdd(src_feature, confident_example, src_label_cls, confident_label)

            # update joint loss function
            optimizer.zero_grad()
            loss = cls_loss # + 0.50*((1-LOSS_WEIGHT)*mmd_loss)       # 0.50*((1-LOSS_WEIGHT)*mmd_loss + LOSS_WEIGHT*cdd_loss)
            loss.backward()
            optimizer.step()

            # calculate the correct
            # calculate the correct
            _, pred = torch.max(src_output_cls, dim=1)
            correct += pred.eq(src_label_cls.data.view_as(pred)).sum()
            count += pred.size(0)
        
        accuracy = float(correct) / count
        
        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/accuracy", accuracy, epoch)

        writer.add_scalar("train/class-loss", cls_loss, epoch)
    

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    
    session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    label_lists = [session1_label, session2_label, session3_label]

    # session1_trial = [42,23,49,32,22,40,38,52,36,42,12,27,54,42,64,35,17,44,35,12,28,28,43,34]
    # session2_trial = [55,25,34,36,53,27,34,46,34,20,60,12,36,27,44,15,46,49,45,10,37,44,24,19]
    # session3_trial = [42,32,23,45,48,26,64,23,26,16,51,41,39,19,28,44,14,17,45,22,39,38,41,39]
    # trial_lists = [session1_trial, session2_trial, session3_trial]

    parser = argparse.ArgumentParser(description='Transfer Learning')
    # parser.add_argument('--method', type=str, nargs='?', default='CDD', help='select the method: CDD or MMD')
    parser.add_argument('--session', type=str, nargs='?', default='1', help='select the session')
    parser.add_argument('--tar_file', type=str, nargs='?', default='1_20160518.mat', help="target file")
    args = parser.parse_args()
    
    # Load data set
    config_path = {"file_path": "/data2/EEG_data/SEED4/eeg_feature_smooth/"+args.session+"/"}
    path_list = get_data_path(config_path["file_path"])
    path_list.remove(config_path["file_path"]+args.tar_file)
    target_path_list = [config_path["file_path"]+args.tar_file]
    source_path_list = path_list # [path_list.pop()]
    
    # Data loader
    source_sample, source_label = load4train(source_path_list, label_lists[int(args.session) - 1])
    target_sample, target_label = load4train(target_path_list, label_lists[int(args.session) - 1])

    source_dset = torch.utils.data.TensorDataset(source_sample, source_label)
    target_dset = torch.utils.data.TensorDataset(target_sample, target_label)
    test_dset = torch.utils.data.TensorDataset(target_sample, target_label)
    
    source_loader = torch.utils.data.DataLoader(source_dset, batch_size=30, shuffle=True, num_workers=4)
    target_loader = torch.utils.data.DataLoader(target_dset, batch_size=30, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=64, shuffle=True, num_workers=4)

    data_loader = {"source_loader": source_loader, "target_loader": target_loader, "test_loader": test_loader}

    # net and optim config
    net_config={"fts":310, "cls":4}
    optim_config = {"lr":1e-3, "weight_decay":0.0005}
    #optim_config = {"lr":1e-3, "momentum":0.9, }
    # Start
    print("The source domain: {}\nthe target domain: {}".format(source_path_list, target_path_list))
    
    writer = SummaryWriter("data/expriment-seed4.70.50/sesion"+args.session +"UDA-none/"+args.tar_file)
    main(data_loader, net_config, optim_config, cuda, writer)
    writer.close()