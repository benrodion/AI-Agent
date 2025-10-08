def expand_with_precomputed_neighbors(
    anchors, 
    document_store, 
    m: int = 2, 
    same_parent_only: bool = True
):
    """
    anchors: list[Document] returned by retriever
    document_store: the same store used by the retriever
    m: max neighbors per anchor to add
    same_parent_only: keep neighbors from same parent doc/section (optional but recommended)
    """

    # Keep anchors by id (preserve order later)
    expanded = {d.id: d for d in anchors}
    anchor_ids = set(expanded.keys())

    # Collect neighbor IDs to fetch (unique)
    want_ids = set()
    for a in anchors:
        nn_ids = (a.meta or {}).get("nn_ids", [])[:m]
        for nid in nn_ids:
            if nid not in anchor_ids:
                want_ids.add(nid)

    if not want_ids:
        return list(anchors)

    # Fetch neighbors by id (API varies slightly across stores)
    try:
        neighbors = document_store.get_documents_by_id(ids=list(want_ids))
    except Exception:
        neighbors = document_store.filter_documents(filters={"id": {"$in": list(want_ids)}})

    # Optional: only keep neighbors from the same parent_id as their *anchor set*
    if same_parent_only:
        # if anchors disagree on parent_id, we skip this constraint
        anchor_parents = { (a.meta or {}).get("parent_id") for a in anchors }
        if len(anchor_parents) == 1:
            only_pid = next(iter(anchor_parents))
            neighbors = [n for n in neighbors if (n.meta or {}).get("parent_id") == only_pid]

    # Merge (don’t override anchors)
    for n in neighbors:
        expanded.setdefault(n.id, n)

    # Keep original anchors first, then the newly added neighbors (stable)
    return anchors + [d for d_id, d in expanded.items() if d_id not in anchor_ids]



from typing import List, Dict, Any, Optional
from haystack import component, Document

@component
class NeighborExpander:
    """
    Expands retriever hits with precomputed neighbors stored in doc.meta['nn_ids'].
    Returns anchors first, then newly added neighbors (deduped).
    """
    def __init__(self, document_store, m: int = 2, same_parent_only: bool = True):
        self.document_store = document_store
        self.default_m = m
        self.default_same_parent_only = same_parent_only
    
    @component.output_types(documents=List[Document])  # Add output type hint
    def run(
        self, 
        documents: List[Document],
        m: Optional[int] = None,  
        same_parent_only: Optional[bool] = None, 
        ) -> Dict[str, Any]:
        if not documents:
            return {"documents": []}
        
        # use runtime params if provided, else fall back to constructor default
        m = self.default_m if m is None else m
        same_parent_only = self.default_same_parent_only if same_parent_only is None else same_parent_only
        
        # Keep anchors in order; fast dedupe set
        expanded_by_id = {d.id: d for d in documents}
        anchor_ids = set(expanded_by_id.keys())
        
        # Collect unique neighbor IDs to fetch
        want_ids = set()
        for a in documents:
            nn_ids = (a.meta or {}).get("nn_ids", [])[:m]
            for nid in nn_ids:
                if nid not in anchor_ids:
                    want_ids.add(nid)
        
        neighbors: List[Document] = []
        if want_ids:
            try:
                neighbors = self.document_store.filter_documents(
                    filters={"field": "id", "operator": "in", "value": list(want_ids)}
                )
            except (AttributeError, NotImplementedError, TypeError):  # More specific
                neighbors = self.document_store.filter_documents(
                    filters={"id": {"$in": list(want_ids)}}
                )
        
        # Optional same-parent filtering
        if same_parent_only:
            anchor_parents = {(a.meta or {}).get("parent_id") for a in documents}
            if len(anchor_parents) == 1:
                only_pid = next(iter(anchor_parents))
                if only_pid is not None:  # Add None check
                    neighbors = [
                        n for n in neighbors 
                        if (n.meta or {}).get("parent_id") == only_pid
                    ]
        
        # Merge neighbors
        for n in neighbors:
            expanded_by_id.setdefault(n.id, n)
        
        # Anchors first, then newly added neighbors
        ordered = documents + [
            d for did, d in expanded_by_id.items() 
            if did not in anchor_ids
        ]
        
        return {"documents": ordered}