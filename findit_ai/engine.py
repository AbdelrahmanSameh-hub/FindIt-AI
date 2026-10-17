"""
FindIt AI - Unified Multimodal Matching Engine
Orchestrates BM25 text search, Color Histogram intersection,
Perceptual Hashing (pHash), and Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any, Tuple
from .algorithms import (
    BM25,
    ColorHistogramMatcher,
    PerceptualHasher,
    ReciprocalRankFusion
)
from .dataset import (
    FOUND_DATABASE,
    CAMPUS_DISTANCES,
    generate_synthetic_image
)

class FindItEngine:
    """
    Unified Intelligent Search Engine for Lost & Found items.
    Combines:
      - Textual search (BM25)
      - Visual Color matching (Histogram Intersection)
      - Shape/Texture matching (pHash with DCT)
      - Spatial & Temporal context optimization (RRF)
    """
    def __init__(self, found_items: List[Dict[str, Any]] = None):
        self.items = found_items if found_items is not None else FOUND_DATABASE
        self.item_dict = {item['id']: item for item in self.items}

        # Initialize Algorithmic Components
        self.bm25 = BM25(k1=1.5, b=0.75)
        self.color_matcher = ColorHistogramMatcher(bins_per_channel=4)
        self.hasher = PerceptualHasher(hash_size=8)
        self.fusion = ReciprocalRankFusion(k_constant=40)

        # Precompute Index & Features
        self._build_indexes()

    def _build_indexes(self):
        """Indexes text corpus and precomputes visual features for all database items."""
        # 1. Text Indexing (Title + Description)
        corpus = [f"{item['title']} {item['description']}" for item in self.items]
        self.bm25.fit(corpus)

        # 2. Visual Features Precomputation
        for item in self.items:
            # Color histogram
            item['color_hist'] = self.color_matcher.compute_histogram(item['pixels_rgb'])
            # Perceptual hash (DCT 64-bit)
            item['phash'] = self.hasher.compute_phash(item['gray_grid'])

    def add_found_item(
        self,
        title: str,
        description: str,
        category: str,
        location: str,
        location_ar: str,
        color_rgb: Tuple[int, int, int],
        pattern: str = 'solid',
        confidential_note: str = "",
        timestamp: float = 0.0
    ) -> Dict[str, Any]:
        """
        Registers a newly found item into the system, precomputes its visual features,
        and dynamically updates the BM25 and multimodal indexes.
        """
        new_id = f"F-{len(self.items) + 101}"
        pixels, gray_grid = generate_synthetic_image(color_rgb, pattern)

        new_item = {
            'id': new_id,
            'title': title,
            'description': description,
            'category': category,
            'location': location,
            'location_ar': location_ar,
            'color': color_rgb,
            'pattern': pattern,
            'confidential_note': confidential_note,
            'timestamp': timestamp,
            'pixels_rgb': pixels,
            'gray_grid': gray_grid
        }

        self.items.append(new_item)
        self.item_dict[new_id] = new_item
        self._build_indexes()
        return new_item

    def search(
        self,
        query_text: str = "",
        query_color: Tuple[int, int, int] = None,
        query_pattern: str = 'solid',
        query_location: str = "Library",
        query_timestamp: float = 12.0,
        weights: Dict[str, float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Executes a multimodal search for a lost item report.
        """
        if weights is None:
            # Balanced multimodal default weights
            weights = {
                'text': 1.0,
                'color': 0.8,
                'phash': 0.6
            }

        # --- Modality 1: BM25 Text Search ---
        text_scores = self.bm25.score(query_text) if query_text else [0.0] * len(self.items)
        ranked_text = sorted(
            [(self.items[i]['id'], text_scores[i]) for i in range(len(self.items))],
            key=lambda x: x[1],
            reverse=True
        )

        # --- Modality 2 & 3: Visual Features (Color & Shape) ---
        if query_color is not None:
            q_pixels, q_gray = generate_synthetic_image(query_color, query_pattern)
            q_hist = self.color_matcher.compute_histogram(q_pixels)
            q_phash = self.hasher.compute_phash(q_gray)

            color_scores = []
            phash_scores = []
            for item in self.items:
                c_sim = self.color_matcher.histogram_intersection(q_hist, item['color_hist'])
                p_sim = self.hasher.similarity(q_phash, item['phash'])
                color_scores.append((item['id'], c_sim))
                phash_scores.append((item['id'], p_sim))

            ranked_color = sorted(color_scores, key=lambda x: x[1], reverse=True)
            ranked_phash = sorted(phash_scores, key=lambda x: x[1], reverse=True)
        else:
            ranked_color = [(item['id'], 0.0) for item in self.items]
            ranked_phash = [(item['id'], 0.0) for item in self.items]

        # --- Modality 4: Reciprocal Rank Fusion + Spatio-Temporal Weighting ---
        ranked_modalities = {
            'text': ranked_text,
            'color': ranked_color,
            'phash': ranked_phash
        }

        fused_results = self.fusion.fuse(
            ranked_lists=ranked_modalities,
            weights=weights,
            item_metadata=self.item_dict,
            query_location=query_location,
            query_timestamp=query_timestamp,
            campus_distances=CAMPUS_DISTANCES
        )

        # Normalize score into a friendly match percentage [0% .. 100%]
        max_possible_rrf = (weights['text'] / 41) + (weights['color'] / 41) + (weights['phash'] / 41)

        output = []
        for item_id, final_score, breakdown in fused_results[:top_k]:
            item = self.item_dict[item_id]
            match_percentage = min(99.0, max(5.0, (final_score / max_possible_rrf) * 100.0))

            output.append({
                'item_id': item_id,
                'title': item['title'],
                'location': item['location_ar'],
                'category': item['category'],
                'match_percentage': round(match_percentage, 1),
                'final_score': round(final_score, 4),
                'breakdown': {
                    'temporal_decay': round(breakdown['temporal'], 2),
                    'spatial_proximity': round(breakdown['spatial'], 2),
                    'text_rank': [idx for idx, (iid, _) in enumerate(ranked_text) if iid == item_id][0] + 1,
                    'color_rank': [idx for idx, (iid, _) in enumerate(ranked_color) if iid == item_id][0] + 1,
                    'phash_rank': [idx for idx, (iid, _) in enumerate(ranked_phash) if iid == item_id][0] + 1,
                },
                'verification_guide': item['confidential_note']
            })

        return output
