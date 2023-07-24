import numpy as np

def evaluate_recall(sims, mode='i2t'):
    if mode == 'i2t':
        recall, _ = i2t(sims, return_ranks=False)
        r1i, r5i, r10i, _, _ = recall
        r1t, r5t, r10t = None, None, None
    if mode == 't2i':
        recall, _ = t2i(sims, return_ranks=False)
        r1t, r5t, r10t, _, _ = recall
        r1i, r5i, r10i = None, None, None
    if mode == 'both':
        recall_i2t, _ = i2t(sims, return_ranks=False)
        recall_t2i, _ = t2i(sims, return_ranks=False)
        r1i, r5i, r10i, _, _ = recall_i2t
        r1t, r5t, r10t, _, _ = recall_t2i
    return r1i, r5i, r10i, r1t, r5t, r10t

def i2t(sims, return_ranks=False):
    # sims (n_imgs, n_caps)
    n_imgs, n_caps = sims.shape
    ranks = np.zeros(n_imgs)
    top1 = np.zeros(n_imgs)
    results = []
    for index in range(n_imgs):
        result = dict()
        result['id'] = index
        inds = np.argsort(sims[index])[::-1]
        result['top5'] = list(inds[:5])
        result['top1'] = inds[0]
        result['top10'] = list(inds[:10])
        result['ranks'] = []
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            result['ranks'].append((i, tmp))
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if rank<1:
            result['is_top1'] = 1
        else:
            result['is_top1'] = 0
        if rank<5:
            result['is_top5'] = 1
        else:
            result['is_top5'] = 0

        results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), results
    else:
        return (r1, r5, r10, medr, meanr), results

def t2i(sims, return_ranks=False):
    # sims (n_imgs, n_caps)
    n_imgs, n_caps = sims.shape
    ranks = np.zeros(5*n_imgs)
    top1 = np.zeros(5*n_imgs)
    # --> (5N(caption), N(image))
    sims = sims.T # ncap, nimg
    results = []
    for index in range(n_imgs):
        for i in range(5):
            result = dict()
            result['id'] = 5*index+i
            inds = np.argsort(sims[5 * index + i])[::-1]
            result['top5'] = list(inds[:5])
            result['top10'] = list(inds[:10])
            result['top1'] = inds[0]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

            if ranks[5*index+i]<1:
                result['is_top1'] = 1
            else:
                result['is_top1'] = 0

            if ranks[5*index+i] <5:
                result['is_top5'] =1
            else:
                result['is_top5'] = 0
            result['ranks'] = [(index, ranks[5*index+i])]
            results.append(result)

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1),results
    else:
        return (r1, r5, r10, medr, meanr), results