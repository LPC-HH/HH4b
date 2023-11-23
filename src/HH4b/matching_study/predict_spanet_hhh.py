import pandas as pd

def remove_elements_with_pd(h_index_char, selected_pairs):
    pairs_pd = pd.DataFrame()
    pairs_pd["pairs_str"] = np.char.mod('%02d', selected_pairs)
    pairs_pd["jet0"] = pairs_pd["pairs_str"].str[0].astype(int)
    pairs_pd["jet1"] = pairs_pd["pairs_str"].str[1].astype(int)

    # just not smart enough to figure this out w/o a loop
    pairs_used = []
    njets = h_index_char.shape[2]
    npairings = h_index_char.shape[3]
    for j in range(njets):
        used_j = []
        for i in range(npairings):
            x = pd.Series(h_index_char[:, 0][:, i]).astype(str)
            used = (
                (x.str[0].astype(int) == pairs_pd["jet0"]) |
                (x.str[1].astype(int) == pairs_pd["jet0"]) |
                (x.str[0].astype(int) == pairs_pd["jet1"]) | 
                (x.str[1].astype(int) == pairs_pd["jet1"])
            )
            used_j.append(used.values)
        used_j = np.array(used_j).T
        pairs_used.append(used_j)
    pairs_used = np.array(pairs_used)
    pairs_used = np.transpose(pairs_used, (1, 0, 2))
    return pairs_used

def get_maximas(assignment_prob):
    """
    Get indices of possible jet pairings (10*(10-1) / 2 = 45) for a given higgs
    sorted by maximum assignment probability
    Jet pairings are an int, e.g. 1 or 12, which should be converted to a string:
    - 1 => 01 pairs
    - 12 => 12 pairs
    """
    # get 10*10 assignment probabilities
    # get upper triangle to avoid pairing repetitions
    assignment_prob_ak = ak.from_numpy(np.triu(assignment_prob[:][:, 0:njets, 0:njets]))
    arr_flat = ak.flatten(assignment_prob_ak, axis=2)
    # sort pairings by maximum assignment probabilities
    max_indices = ak.argsort(arr_flat, ascending=False, axis=1).to_numpy()[:, :45]
    max_values = arr_flat[max_indices]
    return max_indices, max_values

def get_pairs_hhh(assignment_probabilities, detection_probabilities):
    # all possible pairings for h1, h2, h3 sorted by assignment probability
    index_h1, prob_h1 = get_maximas(assignment_probabilities[0])
    index_h2, prob_h2 = get_maximas(assignment_probabilities[1])
    index_h3, prob_h3 = get_maximas(assignment_probabilities[2])
    hIndex = ak.from_numpy(np.stack([index_h1, index_h2, index_h3], axis=1))

    # convert pairings from integer to string
    h_index_1_char = np.char.mod('%02d', index_h1)
    h_index_2_char = np.char.mod('%02d', index_h2)
    h_index_3_char = np.char.mod('%02d', index_h3)
    h_index_char = np.stack([h_index_1_char, h_index_2_char, h_index_3_char], axis=1)

    # detection probability for h1, h2, h3
    h1Det = detection_probabilities[0]
    h2Det = detection_probabilities[1]
    h3Det = detection_probabilities[2]
    hDet = np.stack([h1Det, h2Det, h3Det]).T
    # sort detection probabiilty
    hDetMax = ak.argsort(ak.from_numpy(hDet), ascending=False, axis=1)

    # get pairings of higgs with max detection probability
    higgs_1 = hIndex[hDetMax[:, 0:1]]
    # select the first pairs (sorted by assignment probability)
    higgs_1_pairs = ak.flatten(higgs_1[:, :, 0]).to_numpy()

    # get mask for pairings that are already in use
    is_higgs_1_pair = remove_elements_with_pd(h_index_char, higgs_1_pairs)
    hIndex_wo1 = ak.mask(hIndex, ~is_higgs_1_pair)

    # get pairings of higgs with 2nd max detection probability
    higgs_2 = hIndex_wo1[hDetMax[:, 1:2]]
    # select the first pairs (that are not masked)
    higgs_2_pairs = np.array([h.to_numpy().compressed()[0] for h in higgs_2])

    # get mask for pairings that are already in use
    is_higgs_2_pair = remove_elements_with_pd(h_index_char, higgs_2_pairs)
    hIndex_wo2 = ak.mask(hIndex_wo1, (~is_higgs_2_pair) & (~is_higgs_1_pair))

    # get pairings of higgs with 2nd max detection probability
    higgs_3 = hIndex_wo2[hDetMax[:, 2:3]]
    higgs_3_pairs = np.array([h.to_numpy().compressed()[0] for h in higgs_3])

    return higgs_1_pairs,higgs_2_pairs,higgs_3_pairs

