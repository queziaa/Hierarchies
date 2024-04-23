import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def Dist(v1, v2, p=2, Abso=True):
    temp = v1 - v2
    if Abso:
        temp = torch.abs(temp)
    return torch.sum(temp ** p) ** (1/p)

    
def CosSimilarity(v1, v2,p=2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

def JaccardSimilarity(v1, v2, p=2):
    v1 = v1.flatten().tolist()
    v2 = v2.flatten().tolist()
    return len(set(v1) & set(v2)) / len(set(v1) | set(v2))

def Dot(v1, v2, p=2):
    return torch.sum(v1 * v2)

def Diffnorm(v1, v2, p=2):
    # 计算v1和v2的差的范数，然后将结果转换为PyTorch张量
    return torch.tensor(np.linalg.norm(v1 - v2))

def HyperScore(v1, v2,p=2):
    return 1 - np.tanh(Dot(v1, v2))

#  cos⁡(𝑢⃗,𝑣⃗)  *  ∥𝑣⃗∥ /  ∥𝑢⃗∥
def HyperScore2(v1, v2,p=2):
    return CosSimilarity(v1, v2) * np.linalg.norm(v2) / np.linalg.norm(v1)

# 计算 d = d(v1,v2)  其输出是一个标量
diff_modes = {
    'Dist': Dist,
    # 'CosSimilarity': CosSimilarity,
    # 'JaccardSimilarity': JaccardSimilarity,
    # 'Dot': Dot,
    # 'Diffnorm': Diffnorm,
    # 'HyperScore':HyperScore,
    # 'HyperScore2':HyperScore2
}

# 我有向量a=[1,2,3] b = [4,5,6] 
# 想得到结果out = [1,2,3,4,5,6] out2=[[1,2],[3,4],[5,6]] out3= [[1,2,3],[4,5,6]]

def Diff(v1, v2):
    return v1 - v2

def connect_1(a, b):
    return torch.cat((a, b))

def connect_2(a, b):
    return torch.stack((a, b), dim=1)

def connect_3(a, b):
    return torch.stack((a, b))

# connect_1(a, b)：这个函数使用torch.cat将两个张量在第0维（默认）上连接起来。例如，如果a和b都是形状为[10, 5]的张量，那么结果将是一个形状为[20, 5]的张量。

# connect_2(a, b)：这个函数使用torch.stack将两个张量在新的维度（第1维）上堆叠起来。例如，如果a和b都是形状为[10, 5]的张量，那么结果将是一个形状为[10, 2, 5]的张量。

# connect_3(a, b)：这个函数也使用torch.stack将两个张量在新的维度（默认为第0维）上堆叠起来。例如，如果a和b都是形状为[10, 5]的张量，那么结果将是一个形状为[2, 10, 5]的张量。

def Add(v1, v2):
    return v1 + v2

def Cross(v1, v2):
    return torch.outer(v1.squeeze(), v2.squeeze())

def Mul(v1, v2):
    return v1 * v2

# 计算 v = d(x,y)  其输出是一个向量
vector_modes = {
    'Diff': Diff,
    'connect_1': connect_1,
    'connect_2': connect_2,
    'connect_3': connect_3,
    'Add': Add,
    'Cross': Cross,
    'Mul': Mul
}
def plt_seaborn(embeddings,vectorsMode,diffMode,p=2,Absolute=False,reversal = False,isplt = True,mulmermod = 1):
    print('*************************************mode:',vectorsMode,diffMode,p)
    LEN = int(embeddings.shape[0]/2)
    # torch.Size([4, 1024])
    # 输出LEN^2个差异 将0至LEN-1 对 LEN至2*LEN-1的差异
    # 创建一个LEN^2个差异的矩阵
    if vectorsMode == 'connect_1':
        diff = torch.zeros((LEN*LEN, embeddings.shape[1]*2))
    elif vectorsMode == 'connect_2':
        diff = torch.zeros((LEN*LEN, embeddings.shape[1],2))
    elif vectorsMode == 'connect_3':
        diff = torch.zeros((LEN*LEN, 2,embeddings.shape[1]))
    elif vectorsMode == 'Cross':
        diff = torch.zeros((LEN*LEN, embeddings.shape[1], embeddings.shape[1]))
    else:
        diff = torch.zeros((LEN*LEN, embeddings.shape[1]))
    Lenindex = 0
    
 
    # 对于输入的LEN 个向量，计算它们之间的差异  输出LEN*LEN个差异
    for i in range(LEN):
        for j in range(LEN):
            diff[Lenindex] = vector_modes[vectorsMode](embeddings[i], embeddings[j+LEN])
            # print(vector_modes[vectorsMode](embeddings[i], embeddings[j]).shape)
            # print(diff[Lenindex].shape)
            Lenindex += 1
    # np.savetxt('diff.csv', diff.detach().numpy(), delimiter = ',')
    # print(diff)
    out = torch.zeros((LEN*LEN, LEN*LEN))
    # print("compute diff")
    # 再计算这LEN*LEN个差异之间的差异
    for i in tqdm(range(LEN*LEN)):
        for j in range(LEN*LEN):
            if reversal:
                Itemp = diff[i] * -1
            else:
                Itemp = diff[i]
            
            if Absolute:
                Itemp = torch.abs(Itemp)
                Jtemp = torch.abs(diff[j])
            else:
                Jtemp = diff[j]
            te = diff_modes[diffMode](Itemp, Jtemp, p)
            # te = diff_modes[diffMode](diff[i], diff[j], p)
            # 如果te不是1*1的张量   根据 mulmermod 处理 1为平均 2为乘积
            if type(te) != float and  len(te.shape) != 0:
                if mulmermod == 1:
                    te = torch.mean(te)
                elif mulmermod == 2:
                    te = torch.prod(te)
            out[i][j] = te


    # 将out 保存到文件 csv
    # np.savetxt('out.csv', out.detach().numpy(), delimiter = ',')
    # print("compute out")    
    # 绘制热力图
    nonSample = []
    trSample = []
    for i in range(LEN*LEN):
        for j in range(LEN*LEN):
            if i > j:
                out[i][j] = 0
                continue
            if (i % LEN) == (j % LEN):
                out[i][j] = 0
                continue
            if abs(i - j) < LEN  and (i // LEN) == (j // LEN):
                out[i][j] = 0
                continue
            if i//LEN==i%LEN and j//LEN==j%LEN and i!=j:
                # 绿线交汇点
                # 转化为float再 append
                # trSample.append(out[i][j])
                trSample.append(out[i][j].item())
            else:
                nonSample.append(out[i][j].item())
                # nonSample.append(out[i][j])

    if isplt:
        import matplotlib.pyplot as plt
        import seaborn as sns
        # 如果out有.detach().numpy()方法就调用
        if hasattr(out, 'detach'):
            out = out.detach().numpy()
        sns.heatmap(out, cmap='coolwarm')
        # 在横竖行中 第LEN-1 和 LEN 以及 2*LEN-1 和 2*LEN 之间等等 绘制紫色线
        for i in range(LEN):
            plt.axvline(x=i*LEN + LEN, color='purple')
            plt.axhline(y=i*LEN + LEN, color='purple')

        # 在横竖行中 在第0*LEN+0 ,  1*LEN+1 等等 个方格中绘制绿色
        for i in range(LEN):
            plt.axvline(x=i*LEN + i + 0.5, color='green')
            plt.axhline(y=i*LEN + i + 0.5, color='green')
        
        plt.show()
    return out,nonSample,trSample




def find_ordered_positions(unsorted_values, members_to_find):
    # 如果有成员有.item()就调用.item()方法
    if hasattr(members_to_find[0], 'item'):
        members_to_find = [member.item() for member in members_to_find] 
        unsorted_values = [value.item() for value in unsorted_values]
    unsorted_values = unsorted_values + members_to_find
    # 排序 从小至大
    sorted_values = sorted(unsorted_values)
    # 寻找members_to_find 在 sorted_values 中的位置
    ordered_positions = sorted([sorted_values.index(member) for member in members_to_find])
    return  ordered_positions,[x for x in [x - i for i, x in enumerate(ordered_positions)] if x != 0]


# input_texts = ['先生',
#                "男人",
#                "父亲",
#                "男",
#                "帅哥",
#                "小伙",

               
#                 '女士',
#                "女人",
#                 "母亲 ",
#                "女",
#                "美女",
#                "姑娘"


#                ]