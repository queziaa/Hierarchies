def plt_seaborn(embeddings,LEN):
    import torch
    import torch.nn.functional as F

    # torch.Size([4, 1024])
    # 输出LEN^2个差异 将0至LEN-1 对 LEN至2*LEN-1的差异
    # 创建一个LEN^2个差异的矩阵
    if len(embeddings) != LEN*LEN:
        diff = torch.zeros((LEN*LEN, embeddings.shape[1]))

        Lenindex = 0
        for i in range(LEN):
            for j in range(LEN):
                diff[Lenindex] = embeddings[i] - embeddings[j + LEN]
                Lenindex += 1
        out = torch.zeros((LEN*LEN, LEN*LEN))
        for i in range(LEN*LEN):
            for j in range(LEN*LEN):
                # out[i][j] = F.cosine_similarity(diff[i].unsqueeze(0), diff[j].unsqueeze(0))
                out[i][j] = torch.dist(diff[i], diff[j])
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