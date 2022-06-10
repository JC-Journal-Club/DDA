import numpy as np
import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # 将多个核合并在一起

def coefficient(category_1, category_2, sample1_label, sample2_label):
    cls_bool1 = (sample1_label == category_1)
    cls_bool2 = (sample2_label == category_2)
    total_cls = torch.cat([cls_bool1, cls_bool2], dim=0).int()

    total_coef = torch.ger(total_cls.cpu(), total_cls.cpu()).cuda()
    return total_coef

def lsd(source, target, source_label, target_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    intra_lsd_val = []
    inter_lsd_val = []
    num_class = 3
    for c1 in range(num_class):
        for c2 in range(num_class):
            coef_val = coefficient(c1, c2, source_label, target_label)
            e_ss = torch.div(coef_val[:n, :n] * XX, (coef_val[:n, :n]).sum()+1e-5)
            e_st = torch.div(coef_val[:n, n:] * XY, (coef_val[:n, n:]).sum()+1e-5)
            e_ts = torch.div(coef_val[n:, :n] * YX, (coef_val[n:, :n]).sum()+1e-5)
            e_tt = torch.div(coef_val[n:, n:] * YY, (coef_val[n:, n:]).sum()+1e-5)
            # e_ss[e_ss != e_ss] = 0
            # e_st[e_st != e_st] = 0
            # e_ts[e_ts != e_ts] = 0
            # e_tt[e_tt != e_tt] = 0

            lsd_val = e_ss.sum() + e_tt.sum() - e_st.sum() - e_ts.sum()
            """if lsd_val.is_nan():
                    continue
            else:"""
            if c1 == c2:
                intra_lsd_val.append(lsd_val)
            elif c1 != c2:
                inter_lsd_val.append(lsd_val)
    	
    loss = sum(intra_lsd_val)/len(intra_lsd_val) - sum(inter_lsd_val)/len(inter_lsd_val)
    return loss

