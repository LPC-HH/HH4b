import pandas as pd
import numpy as np
from HH4b.utils import make_vector
import awkward as ak

def build_inputs(events, MIN_PT=20, MIN_FJPT=200, MIN_FJMASS=0):
    """
    Build input array for SPANET inference
    - Not super efficient
    """
    nevents = len(events.ak4JetPt[0])

    # Jets
    njets = 10
    jet_vars = ["PtCorr", "Eta", "SinPhi", "CosPhi", "PNetB", "Mass"]
    arrays = []
    for i in range(njets):
        df = pd.DataFrame(0, index=np.arange(nevents), columns=jet_vars)
        df["PtCorr"] = events.ak4JetPt[i].values
        df["Eta"] = events.ak4JetEta[i].values
        df["SinPhi"] = np.sin(events.ak4JetPhi[i].values)
        df["CosPhi"] = np.cos(events.ak4JetPhi[i].values)
        df["Mass"] = events.ak4JetMass[i].values
        num = (events.ak4JetbtagPNetProbb[i] + events.ak4JetbtagPNetProbbb[i]).values
        den = (
            events.ak4JetbtagPNetProbb[i]
            + events.ak4JetbtagPNetProbbb[i]
            + events.ak4JetbtagPNetProbc[i]
            + events.ak4JetbtagPNetProbcc[i]
            + events.ak4JetbtagPNetProbg[i]
            + events.ak4JetbtagPNetProbuds[i]
        ).values
        pnetb = np.array([-1.0]*nevents, dtype = float)
        pnetb[den > 0] = num[den > 0] / den[den > 0]
        df["PNetB"] = pnetb
        np_arr = df.values.T.astype(np.float32)
        arrays.append(np_arr)

    Jets_data = np.transpose(np.transpose(arrays, (1, 0, 2)))
    Jets_Pt = Jets_data[:, :, 0]
    Jets_mask = Jets_Pt > MIN_PT

    jets = make_vector(events, "ak4Jet")

    # Pair jets together
    Jets_arrays = {}
    Higgs_vars = ["mass", "pt", "eta", "sinphi", "cosphi", "dr"]
    for i in range(njets):
        name = "Jet%s" % i
        Higgs_list = []
        for j in range(1, njets):
            if i == j:
                continue
            if int(j) < int(i):
                continue
            j_i = jets[:, i]
            j_j = jets[:, j]
            jj = j_i + j_j
            df = pd.DataFrame(0, index=np.arange(nevents), columns=Higgs_vars)
            df["mass"] = jj.mass
            df["pt"] = jj.pt
            df["eta"] = jj.eta
            df["sinphi"] = np.sin(jj.phi)
            df["cosphi"] = np.cos(jj.phi)
            df["dr"] = j_i.deltaR(j_j)
            df = df.fillna(0)
            np_arr = df.values.T.astype(np.float32)
            Higgs_list.append(np_arr)
        Jets_arrays[name] = Higgs_list

    Jet_data = {}
    Jet_mask = {}
    for i in range(njets - 1):
        Jet_data[i] = np.transpose(np.transpose(Jets_arrays[f"Jet{i}"], (1, 0, 2)))
        pt = Jet_data[i][:, :, 0]
        Jet_mask[i] = pt > 20


    # FatJets
    boosted_arrays = []
    fatjet_vars = ["Pt", "Eta", "SinPhi", "CosPhi", "PNetXbb", "PNetXjj", "PNetQCD", "Mass"]
    nfatjets = 3
    for i in range(nfatjets):
        df = pd.DataFrame(0, index=np.arange(nevents), columns=fatjet_vars)
        df["Pt"] = events.ak8FatJetPt[i].values
        df["Eta"] = events.ak8FatJetEta[i].values
        df["SinPhi"] = np.sin(events.ak8FatJetPhi[i].values)
        df["CosPhi"] = np.cos(events.ak8FatJetPhi[i].values)
        df["PNetXbb"] = events.ak8FatJetPNetXbb[i].values
        df["PNetXjj"] = events.ak8FatJetPNetXjj[i].values
        df["PNetQCD"] = events.ak8FatJetPNetQCD[i].values
        df["Mass"] = events.ak8FatJetPNetMass[i].values

        np_arr = df.values.T.astype(np.float32)
        boosted_arrays.append(np_arr)

    # Leptons
    lep_arrays = []
    lep_vars = ["Pt", "Eta", "SinPhi", "CosPhi"]
    nleptons = 2
    for i in range(nleptons):
        df = pd.DataFrame(0, index=np.arange(nevents), columns=lep_vars)
        df["Pt"] = events.LeptonPt[i].values
        df["Eta"] = events.LeptonEta[i].values
        df["SinPhi"] = np.sin(events.LeptonPhi[i].values)
        df["CosPhi"] = np.cos(events.LeptonPhi[i].values)

        np_arr = df.values.T.astype(np.float32)
        lep_arrays.append(np_arr)

    tau_arrays = []
    tau_vars = ["Pt", "Eta", "SinPhi", "CosPhi"]
    ntaus = 2
    for i in range(ntaus):
        df = pd.DataFrame(0, index=np.arange(nevents), columns=tau_vars)
        df["Pt"] = events.tauPt[i].values
        df["Eta"] = events.tauEta[i].values
        df["SinPhi"] = np.sin(events.tauPhi[i].values)
        df["CosPhi"] = np.cos(events.tauPhi[i].values)

        np_arr = df.values.T.astype(np.float32)
        tau_arrays.append(np_arr)

    BoostedJets_data = np.transpose(np.transpose(boosted_arrays, (1, 0, 2)))
    BoostedJets_Pt = BoostedJets_data[:, :, 0]
    BoostedJets_Mass = BoostedJets_data[:, :, 7]
    BoostedJets_mask = BoostedJets_Pt > MIN_FJPT
    if MIN_FJMASS > 0:
        BoostedJets_mask = (BoostedJets_Pt > MIN_FJPT) & (BoostedJets_Mass > MIN_FJMASS)

    Leptons_data = np.transpose(np.transpose(lep_arrays, (1, 0, 2)))
    Leptons_Pt = Leptons_data[:, :, 0]
    Leptons_mask = Leptons_Pt > 20

    Taus_data = np.transpose(np.transpose(tau_arrays, (1, 0, 2)))
    Taus_Pt = Taus_data[:, :, 0]
    Taus_mask = Taus_Pt > 20

    met_arrays = [np.array([events.MET_pt.values.squeeze()])]
    MET_data = np.transpose(met_arrays)
    MET_mask = MET_data[:, :, 0] > 0

    ht_arrays = [np.array([events.ht.values.squeeze()])]
    HT_data = np.transpose(ht_arrays)
    HT_mask = HT_data[:, :, 0] > 0

    jet_pair_dict = {}
    for i in range(1, njets):
        jet_pair_dict[f"Jet{i}_data"] = Jet_data[i-1]
        jet_pair_dict[f"Jet{i}_mask"] = Jet_mask[i-1]


    input_dict = {
        "Jets_data": Jets_data,
        "Jets_mask": Jets_mask,
        "BoostedJets_data": BoostedJets_data,
        "BoostedJets_mask": BoostedJets_mask,
        "Leptons_data": Leptons_data,
        "Leptons_mask": Leptons_mask,
        "Taus_data": Taus_data,
        "Taus_mask": Taus_mask,
        "MET_data": MET_data,
        "MET_mask": MET_mask,
        "HT_data": HT_data,
        "HT_mask": HT_mask,
        **jet_pair_dict,
    }

    return input_dict

def remove_elements_with_pd(h_index_char, selected_pairs):
    pairs_pd = pd.DataFrame()
    pairs_pd["pairs_str"] = np.char.mod('%02d', selected_pairs)
    pairs_pd["jet0"] = pairs_pd["pairs_str"].str[0].astype(int)
    pairs_pd["jet1"] = pairs_pd["pairs_str"].str[1].astype(int)

    # just not smart enough to figure this out w/o a loop
    pairs_used = []
    njets = h_index_char.shape[1]
    npairings = h_index_char.shape[2]
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
    njets = 10
    assignment_prob_ak = ak.from_numpy(np.triu(assignment_prob[:][:, 0:njets, 0:njets]))
    arr_flat = ak.flatten(assignment_prob_ak, axis=2)
    # sort pairings by maximum assignment probabilities
    max_indices = ak.argsort(arr_flat, ascending=False, axis=1).to_numpy()[:, :45]
    max_values = arr_flat[max_indices]
    return max_indices, max_values

def get_pairs_hhh_resolved(assignment_probabilities, detection_probabilities):
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

def get_pairs_hhh_boosted(assignment_probabilities):
    bh1 = assignment_probabilities[0]
    bh2 = assignment_probabilities[1]
    bh3 = assignment_probabilities[2]

    # SPANET creates assignment matrices keeping both AK4 and AK8 jets, so 10 + 3 
    # for boosted assignment, we want only AK8 jets, hence I look only at elements 10,11,12 
    boosted_h1 = (ak.from_regular(ak.from_numpy(bh1[:, 10:13])) > 0.5)
    boosted_h2 = (ak.from_regular(ak.from_numpy(bh2[:, 10:13])) > 0.5) & ~boosted_h1
    boosted_h3 = (ak.from_regular(ak.from_numpy(bh3[:, 10:13])) > 0.5) & (~boosted_h2) & (~boosted_h1)

    return boosted_h1,boosted_h2,boosted_h3

def get_pairs_hhh(output_values, events):
    # get jets
    fatjets = make_vector(events, "ak8FatJet", mstring="PNetMass")
    jets = make_vector(events, "ak4Jet")

    # resolved pairs
    higgs_1_pairs,higgs_2_pairs,higgs_3_pairs = get_pairs_hhh_resolved(output_values[0:3],output_values[6:9])
    resolved_higgs = np.stack([higgs_1_pairs, higgs_2_pairs, higgs_3_pairs], axis=1)

    # boosted jets
    boosted_h1,boosted_h2,boosted_h3 = get_pairs_hhh_boosted(output_values[3:6])
    boosted_higgs = np.stack([boosted_h1, boosted_h2, boosted_h3], axis=1)
    is_boosted = (boosted_h1 | boosted_h2 | boosted_h3)

    # get pairings for each higgs (3)
    # if ~is_boosted: transform pairing into string
    higgs_reconstructed_index = ak.from_numpy(np.repeat([[0, 1, 2]], boosted_higgs.to_numpy().shape[0], axis=0))
    higgs_reconstructed_index_fill = ak.where((boosted_h1 | boosted_h2 | boosted_h3), higgs_reconstructed_index, resolved_higgs)

    # get Higgs 4-vector (for resolved, this is a temporary 4-vector)
    higgs_jet_mass = ak.where(is_boosted, fatjets.mass, resolved_higgs)
    higgs_jet_pt = ak.where(is_boosted, fatjets.eta, resolved_higgs)
    higgs_jet_eta = ak.where(is_boosted, fatjets.phi, resolved_higgs)
    higgs_jet_phi = ak.where(is_boosted, fatjets.pt, resolved_higgs)

    # build dataframe
    pairs_pd = pd.DataFrame()
    for i in range(1,4):
        pairs_pd[f"higgs_{i}_rec"] = higgs_reconstructed_index_fill[:, i-1].to_numpy()
        pairs_pd[f"higgs_{i}_isboosted"] = is_boosted[:, i-1].to_numpy()
        pairs_pd[f"higgs_{i}_pairs_str"] = np.char.mod('%02d', resolved_higgs[:,i-1])
        pairs_pd[f"higgs_{i}_jet0"] = pairs_pd[f"higgs_{i}_pairs_str"].str[0].astype(int)
        pairs_pd[f"higgs_{i}_jet1"] = pairs_pd[f"higgs_{i}_pairs_str"].str[1].astype(int)
        pairs_pd[f"higgs_{i}_mass"] = higgs_jet_mass[:, i-1]
        pairs_pd[f"higgs_{i}_pt"] = higgs_jet_pt[:, i-1]
        pairs_pd[f"higgs_{i}_eta"] = higgs_jet_eta[:, i-1]
        pairs_pd[f"higgs_{i}_phi"] = higgs_jet_phi[:, i-1]

    # add information from resolved
    for i in range(1, 4):
        jjs = []
        for j in range(pairs_pd[f"higgs_{i}_jet0"].shape[0]):
            jet_0 = pairs_pd[f"higgs_{i}_jet0"][j]
            jet0 = jets[j, jet_0]
            jet_1 = pairs_pd[f"higgs_{i}_jet1"][j]
            jet1 = jets[j, jet_1]
            jjs.append((jet0 + jet1))
        pairs_pd[f"higgs_resolved_{i}_mass"] = [jj.mass for jj in jjs]
        pairs_pd[f"higgs_resolved_{i}_eta"] = [jj.eta for jj in jjs]
        pairs_pd[f"higgs_resolved_{i}_phi"] = [jj.phi for jj in jjs]
        pairs_pd[f"higgs_resolved_{i}_pt"] = [jj.pt for jj in jjs]

        pairs_pd.loc[~pairs_pd[f"higgs_{i}_isboosted"], f"higgs_{i}_mass"] = pairs_pd[f"higgs_resolved_{i}_mass"]
        pairs_pd.loc[~pairs_pd[f"higgs_{i}_isboosted"], f"higgs_{i}_eta"] = pairs_pd[f"higgs_resolved_{i}_eta"]
        pairs_pd.loc[~pairs_pd[f"higgs_{i}_isboosted"], f"higgs_{i}_phi"] = pairs_pd[f"higgs_resolved_{i}_phi"]
        pairs_pd.loc[~pairs_pd[f"higgs_{i}_isboosted"], f"higgs_{i}_pt"] = pairs_pd[f"higgs_resolved_{i}_pt"]

    columns = []
    for i in range(1,4):
        columns.append(f"higgs_{i}_mass")
        columns.append(f"higgs_{i}_eta")
        columns.append(f"higgs_{i}_phi")
        columns.append(f"higgs_{i}_pt")
        columns.append(f"higgs_{i}_isboosted")

    return pairs_pd[columns]


def get_values_from_output(output_values):
    """
    0-2: assignment prob, jets
    3-5: assignment prob, boosted jets
    6-8: detection prob, jets
    9-11:  detection prob, boosted jets
    12: classification
    """
    classification = output_values[12]
    probs = {
        "hhh": classification[:, 1],
        "qcd": classification[:, 2],
        "tt": classification[:, 3],
        "vjets": classification[:, 4],
        "vv": classification[:, 5],
        "hhh4b2tau": classification[:, 6],
        "hh4b": classification[:, 7],
        "hh2b2tau": classification[:, 8],
    }

    return probs

def get_values_from_assignment_output(output_values):
    """
    Assignment probability labels
    """
    assignment = output_values[12]
    probs = {
        "0bh0h": assignment[:, 0],
        "3bh0h": assignment[:, 1],
        "2bh1h": assignment[:, 2],
        "1bh2h": assignment[:, 3],
        "0bh3h": assignment[:, 4],
        "2bh0h": assignment[:, 5],
        "1bh1h": assignment[:, 6],
        "0bh2h": assignment[:, 7],
        "1bh0h": assignment[:, 8],
        "0bh1h": assignment[:, 9],
    }

    max_probs = np.argmax(np.stack(assignment, axis=1), axis=0)
    return probs, max_probs