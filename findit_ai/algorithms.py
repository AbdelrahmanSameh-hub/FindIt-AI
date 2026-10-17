"""
FindIt AI - Core Algorithms Module
Includes 4 distinct algorithms covering Information Retrieval, Computer Vision,
Pattern Recognition, and Multi-Criteria Ranking & Optimization:

1. Algorithm 1 (IR): BM25 (Best Matching 25) Probabilistic Relevance Ranking
2. Algorithm 2 (CV): Normalized Color Histogram Intersection & Bhattacharyya Distance
3. Algorithm 3 (PR): Perceptual Hashing (pHash) via Discrete Cosine Transform (DCT)
4. Algorithm 4 (Algorithms): Reciprocal Rank Fusion (RRF) with Spatio-Temporal Decay
"""

import math
import re
from typing import List, Dict, Tuple, Any

# =====================================================================
# ALGORITHM 1: BM25 (Information Retrieval)
# Okapi BM25 Probabilistic Ranking Algorithm
# =====================================================================

class BM25:
    """
    Okapi BM25 ranking algorithm for sparse text relevance scoring.
    Formula:
        Score(D, Q) = sum( IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl))) )
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}  # Term -> number of documents containing term
        self.idf = {}
        self.tokenized_docs = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple multilingual (Arabic + English) normalized tokenizer."""
        # Normalize Arabic characters
        text = text.lower()
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'ى', 'ي', text)
        # Extract word tokens
        tokens = re.findall(r'[\w]+', text)
        # Remove common short stopwords
        stopwords = {
            'في', 'من', 'على', 'إلى', 'عن', 'مع', 'هذا', 'هذه', 'تم', 'كان',
            'in', 'on', 'at', 'the', 'a', 'an', 'and', 'or', 'with', 'by'
        }
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def fit(self, corpus: List[str]):
        """Indexes the corpus and calculates IDF values."""
        self.corpus_size = len(corpus)
        self.tokenized_docs = [self.tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        total_length = sum(self.doc_lengths)
        self.avgdl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Calculate Document Frequency (DF)
        self.doc_freqs = {}
        for doc in self.tokenized_docs:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Calculate Inverse Document Frequency (IDF) using Lucene-style formula
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard smoothed IDF: log(1 + (N - df + 0.5) / (df + 0.5))
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> List[float]:
        """Computes BM25 relevance score for each document against the query."""
        query_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size

        for i, doc in enumerate(self.tokenized_docs):
            doc_len = self.doc_lengths[i]
            # Term counts in document
            term_counts = {}
            for t in doc:
                term_counts[t] = term_counts.get(t, 0) + 1

            doc_score = 0.0
            for q_term in query_tokens:
                if q_term not in term_counts:
                    continue
                tf = term_counts[q_term]
                idf = self.idf.get(q_term, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                doc_score += idf * (numerator / denominator)

            scores[i] = doc_score
        return scores


# =====================================================================
# ALGORITHM 2: Normalized Color Histogram Intersection (Computer Vision)
# =====================================================================

class ColorHistogramMatcher:
    """
    Color Histogram Intersection & Bhattacharyya distance for visual color matching.
    Extracts 3D RGB/HSV-like color bins and calculates similarity:
        Intersection(H1, H2) = sum(min(H1_i, H2_i)) / sum(H1_i)
    """
    def __init__(self, bins_per_channel: int = 4):
        self.bins = bins_per_channel
        self.total_bins = bins_per_channel ** 3

    def compute_histogram(self, rgb_pixel_list: List[Tuple[int, int, int]]) -> List[float]:
        """
        Builds a normalized 3D color histogram from a list of RGB pixels.
        """
        hist = [0.0] * self.total_bins
        if not rgb_pixel_list:
            return hist

        bin_width = 256 / self.bins
        for r, g, b in rgb_pixel_list:
            r_idx = min(self.bins - 1, int(r / bin_width))
            g_idx = min(self.bins - 1, int(g / bin_width))
            b_idx = min(self.bins - 1, int(b / bin_width))
            bin_index = r_idx * (self.bins ** 2) + g_idx * self.bins + b_idx
            hist[bin_index] += 1.0

        # L1 Normalization
        total = sum(hist)
        if total > 0:
            hist = [v / total for v in hist]
        return hist

    @staticmethod
    def histogram_intersection(h1: List[float], h2: List[float]) -> float:
        """Computes histogram intersection similarity in range [0.0, 1.0]."""
        if len(h1) != len(h2) or not h1:
            return 0.0
        return sum(min(v1, v2) for v1, v2 in zip(h1, h2))

    @staticmethod
    def bhattacharyya_coefficient(h1: List[float], h2: List[float]) -> float:
        """
        Computes the Bhattacharyya coefficient (measure of overlap between probability distributions).
        rho(H1, H2) = sum( sqrt(H1_i * H2_i) )
        """
        if len(h1) != len(h2) or not h1:
            return 0.0
        return sum(math.sqrt(max(0.0, v1 * v2)) for v1, v2 in zip(h1, h2))


# =====================================================================
# ALGORITHM 3: Perceptual Hashing (pHash) via 2D-DCT (Pattern Recognition)
# =====================================================================

class PerceptualHasher:
    """
    Perceptual Hashing (pHash) using a 2D Discrete Cosine Transform (DCT).
    Extracts low-frequency structural energy of an image, resilient to scale,
    brightness changes, and minor noise.
    Matches using normalized Hamming similarity: 1.0 - (HammingDistance / 64).
    """
    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4):
        self.hash_size = hash_size  # Typically 8 (produces 64-bit hash)
        self.sample_size = hash_size * highfreq_factor  # 32x32 matrix

    def _discrete_cosine_transform_1d(self, vector: List[float]) -> List[float]:
        """Computes 1D-DCT Type II."""
        n = len(vector)
        result = [0.0] * n
        factor = math.pi / (2.0 * n)
        for k in range(n):
            c_k = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
            sum_val = sum(vector[i] * math.cos((2 * i + 1) * k * factor) for i in range(n))
            result[k] = c_k * sum_val
        return result

    def _discrete_cosine_transform_2d(self, matrix: List[List[float]]) -> List[List[float]]:
        """Computes 2D-DCT by row-column decomposition."""
        rows = len(matrix)
        cols = len(matrix[0])

        # Row-wise 1D-DCT
        row_transformed = [self._discrete_cosine_transform_1d(row) for row in matrix]

        # Column-wise 1D-DCT
        col_transformed = [[0.0] * cols for _ in range(rows)]
        for c in range(cols):
            col_vector = [row_transformed[r][c] for r in range(rows)]
            dct_col = self._discrete_cosine_transform_1d(col_vector)
            for r in range(rows):
                col_transformed[r][c] = dct_col[r]

        return col_transformed

    def compute_phash(self, grayscale_grid_32x32: List[List[float]]) -> str:
        """
        Takes a 32x32 2D grayscale array [0..255], computes DCT,
        extracts the top 8x8 low-frequency coefficients, and produces a 64-bit binary string.
        """
        dct = self._discrete_cosine_transform_2d(grayscale_grid_32x32)

        # Extract top 8x8 low frequency block (excluding DC component at [0][0] from median)
        low_freq = []
        for r in range(self.hash_size):
            for c in range(self.hash_size):
                low_freq.append(dct[r][c])

        # Exclude DC term from median calculation
        ac_terms = low_freq[1:]
        median_val = sorted(ac_terms)[len(ac_terms) // 2] if ac_terms else 0.0

        # Construct 64-bit binary hash
        bits = ['1' if val > median_val else '0' for val in low_freq]
        return ''.join(bits)

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Calculates bitwise Hamming distance between two binary hashes."""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    @classmethod
    def similarity(cls, hash1: str, hash2: str) -> float:
        """Calculates normalized perceptual similarity in range [0.0, 1.0]."""
        if len(hash1) != len(hash2) or not hash1:
            return 0.0
        dist = cls.hamming_distance(hash1, hash2)
        return max(0.0, 1.0 - (dist / len(hash1)))


# =====================================================================
# ALGORITHM 4: Reciprocal Rank Fusion (RRF) with Spatio-Temporal Decay
# (Algorithms / Optimization & Multimodal Information Fusion)
# =====================================================================

class ReciprocalRankFusion:
    """
    RRF combines multiple ranked lists (Text BM25, Visual Color, pHash Structure)
    with Spatio-Temporal proximity weighting:

        Score(item) = sum_m [ w_m / (k + Rank_m(item)) ] * Decay_time(dt) * Proximity(loc1, loc2)
    """
    def __init__(self, k_constant: int = 60):
        self.k = k_constant

    @staticmethod
    def temporal_decay(delta_hours: float, half_life_hours: float = 72.0) -> float:
        """
        Exponential temporal decay: events closer in time get higher priority.
        decay = 2^(-delta_hours / half_life)
        """
        return math.pow(2.0, -abs(delta_hours) / half_life_hours)

    @staticmethod
    def spatial_proximity(loc_query: str, loc_item: str, campus_distance_matrix: Dict[Tuple[str, str], float]) -> float:
        """
        Computes spatial proximity factor in range (0, 1].
        """
        if loc_query == loc_item:
            return 1.0
        pair = (loc_query, loc_item)
        rev_pair = (loc_item, loc_query)
        distance_meters = campus_distance_matrix.get(pair, campus_distance_matrix.get(rev_pair, 500.0))
        # Cauchy/Lorentzian decay based on meters: 1 / (1 + distance / 250m)
        return 1.0 / (1.0 + (distance_meters / 250.0))

    def fuse(
        self,
        ranked_lists: Dict[str, List[Tuple[Any, float]]],
        weights: Dict[str, float],
        item_metadata: Dict[Any, Dict[str, Any]],
        query_location: str,
        query_timestamp: float,
        campus_distances: Dict[Tuple[str, str], float]
    ) -> List[Tuple[Any, float, Dict[str, Any]]]:
        """
        Fuses ranked lists using weighted Reciprocal Rank Fusion + Spatio-Temporal adjustments.
        Returns: List of (item_id, final_score, score_breakdown) sorted by final_score descending.
        """
        fused_scores = {}
        breakdowns = {}

        # 1. Compute RRF base scores from each modality
        for modality, ranked_items in ranked_lists.items():
            w = weights.get(modality, 1.0)
            for rank, (item_id, raw_score) in enumerate(ranked_items, start=1):
                rrf_contrib = w / (self.k + rank)
                if item_id not in fused_scores:
                    fused_scores[item_id] = 0.0
                    breakdowns[item_id] = {'modality_contribs': {}, 'temporal': 1.0, 'spatial': 1.0}
                fused_scores[item_id] += rrf_contrib
                breakdowns[item_id]['modality_contribs'][modality] = rrf_contrib

        # 2. Apply Spatio-Temporal Decay Multipliers
        final_rankings = []
        for item_id, base_score in fused_scores.items():
            meta = item_metadata[item_id]

            # Time delta in hours
            delta_hours = abs(query_timestamp - meta.get('timestamp', query_timestamp))
            time_decay = self.temporal_decay(delta_hours, half_life_hours=48.0)

            # Location proximity
            space_prox = self.spatial_proximity(query_location, meta.get('location', ''), campus_distances)

            final_score = base_score * (0.6 + 0.2 * time_decay + 0.2 * space_prox)

            breakdowns[item_id]['temporal'] = time_decay
            breakdowns[item_id]['spatial'] = space_prox
            breakdowns[item_id]['base_rrf'] = base_score
            breakdowns[item_id]['final_score'] = final_score

            final_rankings.append((item_id, final_score, breakdowns[item_id]))

        # Sort descending by final score
        final_rankings.sort(key=lambda x: x[1], reverse=True)
        return final_rankings
