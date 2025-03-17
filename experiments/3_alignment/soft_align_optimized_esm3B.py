# Load libraries
import sys
import os
from collections import defaultdict
import time
from Bio import SeqIO
import pandas as pd
import torch.nn  as nn
import torch
import networkx as nx
from io import StringIO
import numpy as np
from sklearn.decomposition import PCA
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
top_n = 4
min_diagonal_length = 10
max_mismatches = 3
import pickle

# directories
data_dir = "/data/rajan/vog"
pkl_dir = f"{data_dir}/data-pkls"
msa_fasta_dir = f"{data_dir}/fasta"

saved_mod_dir = "saved_mods"
plot_dir = "plots/new"

emb_dir = f"{data_dir}/emb"
esm2_3B_emb_dir = f"{emb_dir}/esm2_3B"
esm2_3B_seq_emb_dir = f"{esm2_3B_emb_dir}/seq"


selected_data = pickle.load(open('sa_selected_data.pkl', 'rb'))

error_seqs = []
device = 'cuda:1'


def get_scores(qid, hid, output_filename, main_emb_dir, layer):
    pt_emb_path = os.path.join(main_emb_dir, 'pre-trained')
    ft_cls_lora_emb_path = os.path.join(main_emb_dir, 'ft_cls_lora')
    ft_mlm_lora_emb_path = os.path.join(main_emb_dir, 'ft_mlm_lora')
    ft_cont_emb_path = os.path.join(main_emb_dir, 'contrastive')
    try:
        seq1_str = selected_data[selected_data.seq_id==qid].seq_str.item()
        seq2_str = selected_data[selected_data.seq_id==hid].seq_str.item()
        if(len(seq1_str)>1024 or len(seq2_str)>1024):
            print('too long - ignoring')
            print('seq1_id:', qid, '\t', 'seq2_id:', hid)
            if qid not in error_seqs:
                error_seqs.append(qid)
            if hid not in error_seqs:
                error_seqs.append(qid)
            return
        
        # pt
        seq1_emb = torch.load(f"{pt_emb_path}/{qid}.pt")["representations"][layer].to(device)
        seq2_emb = torch.load(f"{pt_emb_path}/{hid}.pt")["representations"][layer].to(device)
        # cls-lora
        seq1_ft_cls_lora_emb = torch.load(f'{ft_cls_lora_emb_path}/{qid}.pt')[1:-1].to(device)
        seq2_ft_cls_lora_emb = torch.load(f'{ft_cls_lora_emb_path}/{hid}.pt')[1:-1].to(device)
        # mlm-lora
        seq1_ft_mlm_lora_emb = torch.load(f'{ft_mlm_lora_emb_path}/{qid}.pt')[1:-1].to(device)
        seq2_ft_mlm_lora_emb = torch.load(f'{ft_mlm_lora_emb_path}/{hid}.pt')[1:-1].to(device)
        # cont
        seq1_ft_cont_emb = torch.from_numpy(np.load(f"{ft_cont_emb_path}/{qid}.npy")[1:-1]).to(device) # for contrastive
        seq2_ft_cont_emb = torch.from_numpy(np.load(f"{ft_cont_emb_path}/{hid}.npy")[1:-1]).to(device)

        data = get_data_matrix(seq1_emb, seq2_emb)
        data_ft_cls_lora = get_data_matrix(seq1_ft_cls_lora_emb, seq2_ft_cls_lora_emb)
        data_ft_mlm_lora = get_data_matrix(seq1_ft_mlm_lora_emb, seq2_ft_mlm_lora_emb)
        data_ft_cont = get_data_matrix(seq1_ft_cont_emb, seq2_ft_cont_emb)

        matches = get_matches_new(seq1_str, seq2_str, data)
        matches_ft_cls_lora = get_matches_new(seq1_str, seq2_str, data_ft_cls_lora)
        matches_ft_mlm_lora = get_matches_new(seq1_str, seq2_str, data_ft_mlm_lora)
        matches_ft_cont = get_matches_new(seq1_str, seq2_str, data_ft_cont)

        longest_path = get_longest_path(data, matches)
        longest_path_ft_cls_lora = get_longest_path(data_ft_cls_lora, matches_ft_cls_lora)
        longest_path_ft_mlm_lora = get_longest_path(data_ft_mlm_lora, matches_ft_mlm_lora)
        longest_path_ft_cont = get_longest_path(data_ft_cont, matches_ft_cont)

        score = sum([data.iloc[x] for x in longest_path])
        score_ft_cls_lora = sum([data_ft_cls_lora.iloc[x] for x in longest_path_ft_cls_lora])
        score_ft_mlm_lora = sum([data_ft_mlm_lora.iloc[x] for x in longest_path_ft_mlm_lora])
        score_ft_cont = sum([data_ft_cont.iloc[x] for x in longest_path_ft_cont])

        with open(f"results/{output_filename}", 'a') as f:
            # f.write("\tLen_pt\tLen_ft\tScore_pt\tScore_ft\tSeq1_id\tSeq2_id\tLen_seq1\tLen_seq2\n")
            f.write(f"\t{len(longest_path)}\t{len(longest_path_ft_cls_lora)}\t{len(longest_path_ft_mlm_lora)}\t{len(longest_path_ft_cont)}\t{score}\t{score_ft_cls_lora}\t{score_ft_mlm_lora}\t{score_ft_cont}\t{qid}\t{hid}\t{len(seq1_str)}\t{len(seq2_str)}\n")
    except Exception as e:
        print(e)
        print('seq1_id:', qid, '\t', 'seq2_id:', hid)
        if qid not in error_seqs:
            error_seqs.append(qid)
        if hid not in error_seqs:
            error_seqs.append(qid)
    with open('error_seqs_vog.txt', 'a') as f:
        for seq in error_seqs:
            f.write(f"{seq}\n")
    return


i = 0
def get_data_matrix(seq1_emb, seq2_emb):
    x_tensor = seq1_emb
    y_tensor = seq2_emb

    # Normalize the vectors (this is needed for cosine similarity)
    x_norm = x_tensor / x_tensor.norm(dim=1)[:, None]
    y_norm = y_tensor / y_tensor.norm(dim=1)[:, None]

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = torch.mm(x_norm, y_norm.transpose(0,1))

    # If you need the output as a DataFrame
    data = pd.DataFrame(cosine_similarity_matrix.cpu().numpy())
    return data


def approximate_similarity_matrix(seq1_embeddings, seq2_embeddings, n_components=5):
    """
    Calculates an approximate similarity matrix by reducing the dimensionality of embeddings.
    """
    # Extract the embeddings
    seq1_embeddings_tensor = seq1_embeddings["representations"][36]
    seq2_embeddings_tensor = seq2_embeddings["representations"][36]
    
    # Concatenate the embeddings 
    combined_embeddings = torch.cat((seq1_embeddings_tensor, seq2_embeddings_tensor), dim=0)
    
    # Numpy for PCA
    combined_embeddings_np = combined_embeddings.detach().numpy()  
    
    # Dimension reduction with PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(combined_embeddings_np)
    
    # Separate reduced embeddings
    reduced_seq1_embeddings = reduced_embeddings[:len(seq1_embeddings_tensor)]
    reduced_seq2_embeddings = reduced_embeddings[len(seq1_embeddings_tensor):]
    
    # Standardize reduced embeddings
    # Cosine similarity on reduced embeddings
    norm_seq1 = np.linalg.norm(reduced_seq1_embeddings, axis=1, keepdims=True)
    norm_seq2 = np.linalg.norm(reduced_seq2_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(reduced_seq1_embeddings / norm_seq1, (reduced_seq2_embeddings / norm_seq2).T)
    similarity_matrix = pd.DataFrame(similarity_matrix)
    return similarity_matrix


def find_mutual_matches_optimized(data, top_n = 4):

    # Find the top_n indices for each line
    top_n_indices_rows = np.argsort(-data.values, axis=1)[:, :top_n]
    
    # Find the top_n indices for each column
    top_n_indices_cols = np.argsort(-data.values, axis=0)[:top_n, :]
    
    matches = set()
    for i in range(data.shape[0]):  # For each line
        for j in top_n_indices_rows[i]:  # For each top_n index in the line
            if i in top_n_indices_cols[:, j]:  # If row index is in column top_n
                matches.add((i, j))
                
    return matches


def add_matching_neighbors_optimized(seq_1_str, seq_2_str, matches):
    temp_set = set()
    for match in matches:
        i, j = match
        # Checking the neighbors of each match
        if i > 0 and j > 0 and seq_1_str[i - 1] == seq_2_str[j - 1]:
            temp_set.add((i - 1, j - 1))
        if i < len(seq_1_str) - 1 and j < len(seq_2_str) - 1 and seq_1_str[i + 1] == seq_2_str[j + 1]:
            temp_set.add((i + 1, j + 1))

    return matches.union(temp_set)


def find_exclusive_intervals_optimized(intervals):
    # Sort intervals by starting point, then descending end point
    intervals.sort(key=lambda x: (x[0], -x[1]))
    
    exclusive_intervals = []
    max_end_so_far = -1
    
    for interval in intervals:
        # If the end point of the current interval is greater than max_end_so_far,
        # this means that the interval is not included in any of the preceding intervals
        if interval[1] > max_end_so_far:
            exclusive_intervals.append(interval)
            max_end_so_far = interval[1]
    
    return exclusive_intervals


def find_matches_optimized(s, t, offset_val, matches, k, nb_errors=2):
    found_matches = []

    # Optimization: Run through the sequence once, keeping track of errors and matches    
    start = 0
    while start <= len(s) - k:  # Make sure there are enough characters left for a valid match
        error_count = 0
        match_length = 0
        for i in range(start, len(s)):
            # Check whether current positions match or whether a pre-existing match is recognized
            if s[i] == t[i] or (i, i + offset_val) in matches:
                match_length += 1
            else:
                error_count += 1
                if error_count > nb_errors:
                    #If the number of errors exceeds the authorized threshold, end the current check.
                    break
            
            # Check whether the current length of the valid match exceeds the threshold k
            if match_length >= k:
                found_matches.append((start, i - error_count))
                break

        start += 1  # Move to the next starting position for the next check

    # Filter the intervals found to keep only those that are exclusive
    unique_found_matches = find_exclusive_intervals_optimized(found_matches)

    return unique_found_matches


def get_matches_new(seq_1_str, seq_2_str, data, max_mismatches=3):
    matches = find_mutual_matches_optimized(data)
    matches = add_matching_neighbors_optimized(seq_1_str, seq_2_str, matches)
    valid_segments = find_all_matches_optimized(seq_1_str, seq_2_str, max_mismatches, matches)
    valid_segments = sorted(valid_segments, key=lambda x: x[0][0])
    valid_diagonals = get_valid_diagonals(valid_segments)
    matches = cleanup_matches(matches, valid_diagonals)
    
    return matches


def generate_rrotation(s, t, offset):
    """
    generate_lrotation inputs:
    s = seq_1_str
    t = seq_2_str
    offset = position in sequence where offset occurs

    generate_lrotation function rotates seq_2_str 1 position right
    along corresponding seq_1_str for each iteration and
    returns rotated string.
    """
    # If the offset is larger than the length of the
    # sequence 't', raise an exception.
    if offset >= len(s):
        raise Exception(f"offset {offset} larger than seq length {len(s)}")

    lgaps = '-' * offset

    # Extract a substring from sequence 't' starting from the offset
    # index up to the length of 's'.
    # my_str represents the part of 't' that will be kept after the rotation.
    my_str = t[0:len(s) - offset]

    # Generate a string of '-' characters of length equal to the remaining
    # length of 's' after adding 'my_str'.
    # rgaps represents the right gaps that will be added to the end of the sequence.
    rgaps = '-' * (len(s) - len(lgaps + my_str))

    return lgaps + my_str + rgaps


def generate_lrotation(s, t, offset):
    """
    generate_lrotation inputs:
    s = seq_1_str
    t = seq_2_str
    offset = position in sequence where offset occurs

    generate_lrotation function rotates seq_2_str 1 position left
    along corresponding seq_1_str for each iteration and
    returns rotated string.
    """
    # If the offset is larger than the length of the
    # sequence 't', raise an exception.
    if offset >= len(t):
        raise Exception(f"offset {offset} larger than seq length {len(s)}")

    # Extract a substring from sequence 't' starting from the offset
    # index up to the length of 's'.
    # my_str represents the part of 't' that will be kept after the rotation.
    my_str = t[offset:len(s)]

    # Generate a string of '-' characters of length equal to the remaining
    # length of 's' after adding 'my_str'.
    # rgaps represents the right gaps that will be added to the end of the sequence.
    rgaps = '-' * (len(s) - len(my_str))

    return my_str + rgaps


def find_all_matches_optimized(s, t, k, matched_pairs):
    """
    find_all_matches inputs:
    s = seq_1 sequence string denoted as 'seq_1_str'
    t = seq_2 sequence string denoted as 'seq_2_str'
    k = max_mismatches, hyperparameter defined above for amount of
    mismatches allowed.
    matched_pairs = current 'matches' list, which contains mutual matches
    and matching neighbors.
    """
    all_matches = []

    # In each iteration, generate a right rotation of 'seq_2_str' by the
    # current index and run find_match function to identify matching pairs
    # in 'seq_1_str' and 'seq_2_str' after rotation.
    # Matched pairs identified during rotation are added to all_matches
    # list.
    for i in range(0, len(s)):
        t_offset = generate_rrotation(s, t, i)

        match_in_i = find_matches_optimized(s, t_offset, -i, matched_pairs, k)

        # Adds another match along the same diagonal to match_in_i
        match_in_j = [(x - i, y - i) for x, y in match_in_i]

        # Adds both matches along same diagonal to 'all_matches' list
        all_matches.extend(list(zip(match_in_i, match_in_j)))

    # In each iteration, generate a left rotation of 'seq_2_str' by the
    # current index and run find_match function to identify matching pairs
    # in 'seq_1_str' and 'seq_2_str' after rotation.
    # Matched pairs identified during rotation are added to all_matches
    # list.
    for i in range(1, len(t)):
        t_offset = generate_lrotation(s, t, i)

        match_in_i = find_matches_optimized(s, t_offset, +i, matched_pairs, k)

        # Adds another match along the same diagonal to match_in_i
        match_in_j = [(x + i, y + i) for x, y in match_in_i]

        # Adds both matches along same diagonal to 'all_matches' list
        all_matches.extend(list(zip(match_in_i, match_in_j)))

    return all_matches


def build_paths_graph(data, matches):
    """
    build_paths_graph function identifies diagonal segments
    from sorted matches.
    """
    dag = {}

    graph = nx.DiGraph()

    max_depth = max([x[0] for x in matches])

    # Sort the matches based on the second element of the match pairs.
    sorted_matches = sorted(matches, key=lambda x: x[1])

    # Loop over the sorted matches and
    # add edges between them to build the graph.
    for i in range(len(sorted_matches) - 1):
        last_depth = max_depth
        dag[sorted_matches[i]] = []

        for j in range(i + 1, len(sorted_matches)):

            if (sorted_matches[i][0] == sorted_matches[j][0]) or (sorted_matches[i][1] == sorted_matches[j][1]):
                # Don't consider overlapping cells
                continue

            if (sorted_matches[j][0]) < last_depth and (sorted_matches[j][0] > sorted_matches[i][0]):
                dag[sorted_matches[i]].append(sorted_matches[j])
                seq_1_idx, seq_2_idx = sorted_matches[j]
                graph.add_edge(sorted_matches[i], sorted_matches[j], weigth=data.iloc[seq_1_idx, seq_2_idx])
                last_depth = sorted_matches[j][0]

    return graph


def get_valid_diagonals(valid_segments):
    """
    valid_segments = sorted(valid_segments)

    get_valid_diagonals function identifies matches that occur consecutively
    in a diagonal and stores them in a dictionary 'valid_diagonals'.
    """
    valid_diagonals = defaultdict(int)

    # Loop over the valid segments and add the length of each segment
    # to its corresponding diagonal in the dictionary.
    for x in valid_segments:
        min_val = min(x[0][0], x[1][0])
        diag = (x[0][0] - min_val, x[1][0] - min_val)
        valid_diagonals[diag] += x[0][1] - x[0][0] + 1

    return valid_diagonals


def cleanup_matches(matches, valid_diagonals):
    """
    cleanup_matches removes matches that do not occur in a valid_diagonal
    but are shorter than min_diagonal_length (hyperparameter).
    """
    remove_elems = []

    # Loop over the matches and add any invalid match to the removal list
    for x in matches:
        min_val = min(x[0], x[1])
        diag = (x[0] - min_val, x[1] - min_val)
        if valid_diagonals[diag] < min_diagonal_length:
            remove_elems.append(x)

    # Remove the invalid matches from the original list
    matches = list(set(matches).difference(remove_elems))

    return matches


def get_longest_path(data, matches):
    longest_path = []

    # If there are any matches left, build a paths graph and find the longest path in the graph
    if len(matches) > 0:
        graph = build_paths_graph(data, matches)
        longest_path = nx.dag_longest_path(graph)

    return longest_path


def soft_align_time_new(seq_1_str, seq_2_str, seq_1_embedding, seq_2_embedding, pca_matrix = False): 
    if pca_matrix :
        data = approximate_similarity_matrix(seq_1_embedding,seq_2_embedding)
    else : 
        data = get_data_matrix(seq_1_embedding, seq_2_embedding)
    matches = get_matches_new(seq_1_str, seq_2_str, data)
    longest_path = get_longest_path(data, matches)
    
    return longest_path

qid = str(sys.argv[1])
hid = str(sys.argv[2])
output_filename = str(sys.argv[3])
get_scores(qid, hid, output_filename, esm2_3B_seq_emb_dir, layer=36)


