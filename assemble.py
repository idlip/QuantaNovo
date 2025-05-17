import math
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# for dwave system simulation
import dimod
import minorminer
from dwave.embedding import embed_ising, unembed_sampleset
from dwave.embedding.utils import edgelist_to_adjacency
from dwave.embedding.chain_breaks import majority_vote

# for qiskit package
# many modules change alot with version
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.visualization import plot_histogram, plot_bloch_vector, plot_bloch_multivector
from qiskit.quantum_info import SparsePauliOp
# from qiskit_aer import AerSimulator

# Optional imports
# import dwave_networkx as dnx
# from dwave.cloud import Client
# from dwave.system.samplers import DWaveSampler


"""
Overlap between pair-wise reads

def align(read1: str, read2: str, mm: int):
    l1, l2 = len(read1), len(read2)
    for shift in range(l1 - l2, l1):
        mmr = 0
        r2i = 0
        for r1i in range(shift, l1):
            if read1[r1i] != read2[r2i]:
                mmr += 1
                r2i += 1
            if mmr > mm:
                break
        if mmr <= mm:
            return l2 - shift
    return 0
"""

def calculate_read_overlap(read1: str, read2: str, max_mismatches: int = 1) -> int:
    """
    Calculate the overlap length between two sequence reads with allowed mismatches.
    
    This function finds the maximum overlap between the end of read1 and the start of read2,
    allowing for a specified number of mismatches.
    
    Args:
        read1 (str): First sequence read
        read2 (str): Second sequence read
        max_mismatches (int): Maximum number of mismatches allowed in the overlap region
        
    Returns:
        int: Length of the overlap between reads. Returns 0 if no valid overlap is found.
        
    Examples:
        >>> calculate_read_overlap("ACTGGT", "TGGTCC", 0)
        4  # Perfect overlap of 'TGGT'
        >>> calculate_read_overlap("ACTGGT", "TGGTCC", 1)
        4  # Overlap allowing 1 mismatch
    """
    # Input validation
    if not isinstance(read1, str) or not isinstance(read2, str):
        raise TypeError("Reads must be strings")
    if not read1 or not read2:
        return 0
    if max_mismatches < 0:
        raise ValueError("Maximum mismatches must be non-negative")
    
    len1, len2 = len(read1), len(read2)
    max_overlap = min(len1, len2)
    
    # Try different overlap positions
    for shift in range(max(0, len1 - len2), len1):
        mismatches = 0
        read2_pos = 0
        overlap_valid = True
        
        # Check for mismatches in the overlapping region
        for read1_pos in range(shift, len1):
            if read1_pos >= len1 or read2_pos >= len2:
                break
                
            if read1[read1_pos] != read2[read2_pos]:
                mismatches += 1
                if mismatches > max_mismatches:
                    overlap_valid = False
                    break
            
            read2_pos += 1
            
        if overlap_valid:
            return len2 - shift
            
    return 0


"""
Convert set of reads to adjacency matrix of pair-wise overlap for TSP

def reads_to_tspAdjM(reads, max_mismatch=0):
    n_reads = len(reads)
    O_matrix = np.zeros((n_reads, n_reads))  # edge directivity = (row id, col id)
    for r1 in range(0, n_reads):
        for r2 in range(0, n_reads):
            if r1 != r2:
                O_matrix[r1][r2] = align(reads[r1], reads[r2], max_mismatch)
                O_matrix = O_matrix / np.linalg.norm(O_matrix)
    return O_matrix
"""

def reads_to_tsp_adj_matrix(reads: str, max_mismatch: int =0):
    """
    Convert a set of reads to an adjacency matrix representing pairwise overlaps for TSP.
    
    Args:
        reads (list of str): List of read strings.
        max_mismatch (int): Maximum allowed mismatches for overlap.
    
    Returns:
        np.ndarray: Normalized adjacency matrix of overlaps.
    """
    n_reads = len(reads)
    overlap_matrix = np.zeros((n_reads, n_reads))
    
    for i in range(n_reads):
        for j in range(n_reads):
            if i != j:
                overlap_matrix[i, j] = align(reads[i], reads[j], max_mismatch)
    
    norm = np.linalg.norm(overlap_matrix)
    if norm > 0:
        overlap_matrix /= norm
    
    return overlap_matrix


"""
Convert adjacency matrix of pair-wise overlap for TSP to QUBO matrix of TSP

def tspAdjM_to_quboAdjM(tspAdjM, p0, p1, p2):
    n_reads = len(tspAdjM)
    # Initialize
    Q_matrix = np.memmap('large_array.dat', dtype=np.float32, mode='w+', shape=(n_reads**2, n_reads**2))
    # Q_matrix = np.zeros((n_reads**2, n_reads**2))
    # Qubit index semantics: {c(0)t(0) |..| c(i)-t(j) | c(i)t(j+1) |..| c(i)t(n-1) | c(i+1)t(0) |..| c(n-1)t(n-1)}
    # Assignment reward (self-bias)
    p0 = -1.6
    for ct in range(0, n_reads**2):
        Q_matrix[ct][ct] += p0
        # Multi-location penalty
        p1 = -p0  # fixed emperically by trail-and-error
    for c in range(0, n_reads):
        for t1 in range(0, n_reads):
            for t2 in range(0, n_reads):
                if t1 != t2:
                    Q_matrix[c * n_reads + t1][c * n_reads + t2] += p1
                    # Visit repetation penalty
                    p2 = p1
    for t in range(0, n_reads):
        for c1 in range(0, n_reads):
            for c2 in range(0, n_reads):
                if c1 != c2:
                    Q_matrix[c1 * n_reads + t][c2 * n_reads + t] += p2
                    # Path cost
                    # kron of tspAdjM and a shifted diagonal matrix
    for ci in range(0, n_reads):
        for cj in range(0, n_reads):
            for ti in range(0, n_reads):
                tj = (ti + 1) % n_reads
                Q_matrix[ci * n_reads + ti][cj * n_reads + tj] += -tspAdjM[ci][cj]
    print(Q_matrix)
    return Q_matrix
"""

def tspAdjM_to_quboAdjM(tspAdjM, p0, p1, p2):
    """
    Convert a TSP adjacency matrix to a QUBO adjacency matrix.

    Args:
        tspAdjM (np.ndarray): TSP adjacency (distance/overlap) matrix, shape (n, n).
        p0 (float): Assignment reward (self-bias).
        p1 (float): Multi-location penalty.
        p2 (float): Visit repetition penalty.

    Returns:
        np.ndarray: QUBO adjacency matrix, shape (n^2, n^2).
    """
    n = len(tspAdjM)
    N = n * n
    Q = np.zeros((N, N), dtype=np.float32)

    # Assignment reward (self-bias)
    for ct in range(N):
        Q[ct, ct] += p0

    # Multi-location penalty: Each city assigned to only one position in the tour
    for c in range(n):
        for t1 in range(n):
            for t2 in range(n):
                if t1 != t2:
                    i = c * n + t1
                    j = c * n + t2
                    Q[i, j] += p1

    # Visit repetition penalty: Each position in the tour assigned to only one city
    for t in range(n):
        for c1 in range(n):
            for c2 in range(n):
                if c1 != c2:
                    i = c1 * n + t
                    j = c2 * n + t
                    Q[i, j] += p2

    # Path cost: Add TSP adjacency weights for consecutive cities in the tour
    for ci in range(n):
        for cj in range(n):
            for ti in range(n):
                tj = (ti + 1) % n
                i = ci * n + ti
                j = cj * n + tj
                Q[i, j] += -tspAdjM[ci, cj]

    return Q


"""
Convert QUBO matrix of TSP to QUBO dictionary of weighted adjacency list
"""


def quboAdjM_to_quboDict(Q_matrix):
    n_reads = int(math.sqrt(len(Q_matrix)))
    Q = {}
    for i in range(0, n_reads**2):
        ni = "n" + str(int(i / n_reads)) + "t" + str(int(i % n_reads))
        for j in range(0, n_reads**2):
            nj = "n" + str(int(j / n_reads)) + "t" + str(int(j % n_reads))
            if Q_matrix[i][j] != 0:
                Q[(ni, nj)] = Q_matrix[i][j]
    print(Q)
    return Q


"""
Solve a QUBO model using dimod exact solver
"""


def solve_qubo_exact(Q, all=False):
    solver = dimod.ExactSolver()
    response = solver.sample_qubo(Q)
    minE = min(response.data(["sample", "energy"]), key=lambda x: x[1])
    for sample, energy in response.data(["sample", "energy"]):
        if all or energy == minE[1]:
            print(sample)


"""
Solve an Ising model using dimod exact solver
"""


def solve_ising_exact(hii, Jij, plotIt=False):
    solver = dimod.ExactSolver()
    response = solver.sample_ising(hii, Jij)
    print("Minimum Energy Configurations\t===>")
    minE = min(response.data(["sample", "energy"]), key=lambda x: x[1])
    for sample, energy in response.data(["sample", "energy"]):
        if energy == minE[1]:
            print(sample, energy)
    if plotIt:
        y = []
        for sample, energy in response.data(["sample", "energy"]):
            y.append(energy)
        plt.plot(y)
        plt.xlabel("Solution landscape")
        plt.ylabel("Energy")
        plt.savefig("ising.png")
        plt.show()
    # print(hii)
    # print(Jij)

def solve_ising_dwave(hii,Jij):
	config_file='QA_DeNovoAsb/dwcloud.conf'
	client = Client.from_config(config_file, profile='aritra')
	solver = client.get_solver() # Available QPUs: DW_2000Q_2_1 (2038 qubits), DW_2000Q_5 (2030 qubits)
	dwsampler = DWaveSampler(config_file=config_file)

	edgelist = solver.edges
	adjdict = edgelist_to_adjacency(edgelist)
	embed = minorminer.find_embedding(Jij.keys(),edgelist)
	[h_qpu, j_qpu] = embed_ising(hii, Jij, embed, adjdict)

	response_qpt = dwsampler.sample_ising(h_qpu, j_qpu, num_reads=solver.max_num_reads())
	client.close()

	bqm = dimod.BinaryQuadraticModel.from_ising(hii, Jij)
	unembedded = unembed_sampleset(response_qpt, embed, bqm, chain_break_method=majority_vote)
	print("Maximum Sampled Configurations from D-Wave\t===>")
	solnsMaxSample = sorted(unembedded.record,key=lambda x: -x[2])
	for i in range(0,10):
		print(solnsMaxSample[i])
	print("Minimum Energy Configurations from D-Wave\t===>")
	solnsMinEnergy = sorted(unembedded.record,key=lambda x: +x[1])
	for i in range(0,10):
		print(solnsMinEnergy[i])

def ising_solver(hii, Jij):
    solver = dimod.SimulatedAnnealingSampler()
    # edgelist = solver.edges
    # adjdict = edgelist_to_adjacency(edgelist)
    # embed = minorminer.find_embedding(Jij.keys(),edgelist)
    # [h_qpu, j_qpu] = embed_ising(hii, Jij, embed, adjdict)

    bqm = dimod.BinaryQuadraticModel.from_ising(hii, Jij)
    print(bqm)
    sampler = solver.sample(bqm)
    print(sampler)
    print(type(sampler))

    for sample, energy in sampler.data(fields=['sample', 'energy']):
        print(sample, energy)


# r1 = "NAACCTCTCTGTTTACTGATAAGTTCCAGATCCTCCTGGCAACTTGCACAAGTCCGACAACCCTGAACGACCAGGCGTCTTCGTTCATCTATCGGATCTCCACACTCACAACAATGAGTGGCAGATATAGCCTGGTGGTTCAGGCGGCGCA"
# r2 = "NGCACGGATGCTACACGAACCTGATGAACAAACTGGATACGATTGGATTCGACAACAAAAAAGAGATCGGAAGAGCACACGTCTGAACTCCAGTCACACTTGAATCTCGTATGCCGTCTTCTGCTTGAAAAAAAAAACACTTTTCAGCTAC"
# r3 = "NGGATTGTCGGGAGTATCGGCAGCGCCATTGGCGGGGCTGGTTGTGGGGGCGCCTCCCCGCCCCGCGGGACAACCCCTCAGGCCCCCGCGGCGGAAATTCCTTTTTTTAACCGAGGGGTTTACTGGACCCGGATGTGGGCTTTTCCACCAC"
# r4 = "NTGGACATGGATACCCCGTGAGTTACCCGGCGGGCGCGCCTCGTTCATTCACGTTTTTGAACCCGTGGAGGACGGGCAGACTCGCGGTGCAAATGTGTTTTACAGCGTGATGGAGCAGATGAAGATGCTCGACACGCTGCAGAACACGCAG"

r1 = "ATGCGTG"
r2 = "CGTAGCA"
r3 = "ACTTCAG"
r4 = "CAGCTAG"
r5 = "CGATCAG"
r6 = "TGTGCAA"
r7 = "CAGCTAG"

def deNovo_locally():
    reads = [ r1, r2, r3, r4 ]
    tspAdjM = reads_to_tspAdjM(reads)
    quboAdjM = tspAdjM_to_quboAdjM(tspAdjM, -1.6, 1.6, 1.6)
    quboDict = quboAdjM_to_quboDict(quboAdjM)
    hii, Jij, offset = dimod.qubo_to_ising(quboDict)
    solve_ising_exact(hii, Jij, plotIt=True)
    # solve_ising_dwave(hii,Jij)
    ising_solve(hii, Jij)
    print(f"Reads used:\n{reads}")

deNovo_locally()
