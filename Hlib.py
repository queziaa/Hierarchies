import torch
import torch.nn.functional as F
import numpy as np




def Dist(v1, v2, p=2,Absolute = 1):
    temp = v1 - v2
    sum = 0
    for i in range(len(temp)):
        if Absolute:
            temp[i] = abs(temp[i])
        temp[i] = temp[i] ** p
        sum += temp[i]
    return sum ** (1/p)

    
def CosSimilarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))

def JaccardSimilarity(v1, v2):
    v1 = v1.to('cpu').numpy()
    v2 = v2.to('cpu').numpy()
    return len(set(v1) & set(v2)) / len(set(v1) | set(v2))

# 向量点乘
def Dot(v1, v2):
    return v1.dot(v2)


# 计算 d = d(v1,v2)  其输出是一个标量
diff_modes = {
    'Dist': Dist,
    'CosSimilarity': CosSimilarity,
    'JaccardSimilarity': JaccardSimilarity,
    'Dot': Dot
}

# 我有向量a=[1,2,3] b = [4,5,6] 
# 想得到结果out = [1,2,3,4,5,6] out2=[[1,2],[3,4],[5,6]] out3= [[1,2,3],[4,5,6]]

def Diff(v1, v2):
    return v1 - v2

def connect_1(a, b):
    return np.concatenate((a, b))

def connect_2(a, b):
    return np.array(list(zip(a, b)))

def connect_3(a, b):
    return np.array([a, b])

def Add(v1, v2):
    return v1 + v2

# 向量叉乘
def Cross(v1, v2):
    return np.cross(v1, v2)

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
def plt_seaborn(embeddings,LEN,diffMode,vectorsMode,p=2,Absolute=1,reversal = 1):

    # torch.Size([4, 1024])
    # 输出LEN^2个差异 将0至LEN-1 对 LEN至2*LEN-1的差异
    # 创建一个LEN^2个差异的矩阵
    if len(embeddings) != LEN*LEN:
        diff = torch.zeros((LEN*LEN, embeddings.shape[1]))

        Lenindex = 0
        for i in range(LEN):
            for j in range(LEN):
                if diffMode in diff_modes:
                    diff[Lenindex] = vector_modes[vectorsMode](embeddings[i], embeddings[j], p, Absolute)
                else:
                    # 错误
                    raise ValueError('diffMode Error!')
                Lenindex += 1
        out = torch.zeros((LEN*LEN, LEN*LEN))
        for i in range(LEN*LEN):
            for j in range(LEN*LEN):
                # out[i][j] = F.cosine_similarity(diff[i].unsqueeze(0), diff[j].unsqueeze(0))
                # out[i][j] = torch.dist(diff[i], diff[j])
                out[i][j] = diff_modes[diffMode](diff[i], diff[j], p, Absolute)
    else:
        out = embeddings
    
    # 绘制热力图
    nonSample = []
    trSample = []
    for i in range(LEN*LEN):
        for j in range(LEN*LEN):
            if i > j:
                out[i][j] = -0.5
                continue
            if (i % LEN) == (j % LEN):
                out[i][j] = -0.5
                continue
            if abs(i - j) < LEN  and (i // LEN) == (j // LEN):
                out[i][j] = -0.5
                continue
            if i//LEN==i%LEN and j//LEN==j%LEN and i!=j:
                # <class 'torch.Tensor'> 转化为float
                trSample.append(out[i][j])
            else:
                nonSample.append(out[i][j])


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
    ordered_positions = [sorted_values.index(member) for member in members_to_find]
    print(ordered_positions)
    return [x for x in [x - i for i, x in enumerate(sorted(ordered_positions))] if x != 0]


input_texts = ['先生',
               "男人",
               "父亲",
               "男",
               "帅哥",
               "小伙",

               
                '女士',
               "女人",
                "母亲 ",
               "女",
               "美女",
               "姑娘"


               ]