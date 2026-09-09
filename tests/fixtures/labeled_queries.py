"""Hand-curated labeled query set for dense-vs-hybrid retrieval evaluation.

Ground truth was assigned by reading actual chunk text pulled from the
local database (75 chunks across 2 papers: LiDAR 3D object detection
[pdf_asset 0e541beb] and Energy-Gated Attention / Wavelet PE
[pdf_asset 39dc0970]).

REVIEW REQUIRED: confirm or correct the relevant_chunk_ids for each query
below against your own reading of the source PDFs before treating this as
ground truth. Chunk ids are text_chunk_id values from TextChunkRepository.
"""

from __future__ import annotations

from academic_literature_rag.services.retrieval_evaluation_service import (
    LabeledQuery,
)

LABELED_QUERIES: list[LabeledQuery] = [
    # --- Paper A: LiDAR 3D video object detection ---
    LabeledQuery.create(
        query="What is PMPNet and what problem does it solve?",
        relevant_chunk_ids=[
            "dcb6d210-0322-4b0a-9338-4f54631dd921",  # iterative message passing, receptive field
            "089d3748-76ef-4f8d-9b52-12646e7d6ac3",  # aggregating messages, PMPNet mines geometric relations
        ],
    ),
    LabeledQuery.create(
        query="How does the AST-GRU module capture temporal information across frames?",
        relevant_chunk_ids=[
            "8a237e70-9003-441d-b990-57bc33453fca",  # graph-based message passing -> AST-GRU
            "2dac3cbe-2203-415b-ba84-d00740dd3fac",  # variant of conventional GRU, convolution ops
            "559e4d03-6412-4c82-80e8-40c412a1aa7b",  # STA designed as intra-attention
        ],
    ),
    LabeledQuery.create(
        query="What dataset and benchmark were used to evaluate the detection model?",
        relevant_chunk_ids=[
            "20f6325b-438a-4147-af75-c269917c48a9",  # Method Car Pedestrian Bus ... nuScenes-style table
            "7d6efe4f-257f-4bab-8ab7-70f3c5eebd0b",  # nuScenes benchmark, L1 loss, Experimental Results
        ],
    ),
    LabeledQuery.create(
        query="What is the effect of using longer input sequence lengths on detection accuracy?",
        relevant_chunk_ids=[
            "563d4b3a-33f3-483e-8ac4-7c3d33e6ea14",  # Ablation study for input lengths, T=4, T=5 results
        ],
    ),
    LabeledQuery.create(
        query="What are the limitations of single-frame 3D object detectors compared to video-based approaches?",
        relevant_chunk_ids=[
            "885f90b0-7d91-4dcf-a9ae-856181059f3e",  # typical single-frame detector limitations
            "b47ba3ec-4eef-41ab-8eb7-f81c4fe294c8",  # cannot deal with long-term multi-frame sequences
        ],
    ),
    # --- Paper B: Energy-Gated Attention / Wavelet Positional Encoding ---
    LabeledQuery.create(
        query="What is energy-gated attention (EGA) and how does it modify standard attention?",
        relevant_chunk_ids=[
            "925ccd67-ebd4-4e6d-b0e0-00f7e8e4b112",  # gating value aggregation with learned energy estimate
            "e9839659-10b0-4b8f-9ca2-350036e576f3",  # EGA is most precisely a learned energy gate
        ],
    ),
    LabeledQuery.create(
        query="Are EGA and MOPE complementary, and what results show this?",
        relevant_chunk_ids=[
            "0cf0a31e-5a40-4c2a-8b59-408f4fbf048c",  # improvement over baseline, MOPE alone below baseline
            "47d15d2e-bb5e-450e-a679-ccdceb863c1f",  # not a substitute for the other, more complete mechanism
            "ef0ea416-34b1-40eb-871c-77e36333470e",  # together they achieve +0.119, complementarity supported
        ],
    ),
    LabeledQuery.create(
        query="How does wavelet positional encoding relate to rotary position embeddings (RoPE)?",
        relevant_chunk_ids=[
            "91de36aa-44f0-4ce4-86b3-41e9bc83cc7a",  # connection to prior PE methods, ROPE recovers sinusoidal
            "5a928a7d-2a1f-44bc-8d7b-93498b1d1c34",  # cosine encodes relative position, same as ROPE's rotation angle
        ],
    ),
    LabeledQuery.create(
        query="What dataset was used to evaluate the attention mechanism experiments?",
        relevant_chunk_ids=[
            "3757291b-e933-4e43-9b1f-c477d0ac3f8f",  # Table 1: Main results on TinyShakespeare
        ],
    ),
    LabeledQuery.create(
        query="What practical GPU memory issue was encountered during training, and how was it resolved?",
        relevant_chunk_ids=[
            "f7cd4a70-2160-495b-bd4a-e81c7914c116",  # solved out-of-memory failure at L=16 on 15.6 GB T4 GPU
            "3068ba99-ba0a-456d-a830-46835b679601",  # same fix, followed by Phase 4 results table
        ],
    ),
]
