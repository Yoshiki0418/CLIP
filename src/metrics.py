def compute_top_k_accuracy(similarity_matrix, labels, k=5):
    """
    Top-k Accuracyの計算
    Args:
        similarity_matrix (torch.Tensor): 画像とテキスト埋め込み間の類似度行列 [batch_size, batch_size]
        labels (torch.Tensor): 正解ラベル [batch_size]
        k (int): 上位k個で正解を確認
    Returns:
        float: Top-k Accuracy
    """
    top_k = similarity_matrix.topk(k, dim=-1).indices  # 類似度上位k個のインデックス
    correct = top_k.eq(labels.unsqueeze(1)).any(dim=-1)  # 上位kに正解が含まれるか
    accuracy = correct.float().mean().item()
    return accuracy

def calc_acc(preds):
    return 100. * sum(preds) / len(preds)