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
    # Keep original order for anchors
    expanded = {d.id: d for d in anchors}

    # Collect neighbor IDs to fetch
    want_ids = []
    for a in anchors:
        nn_ids = a.meta.get("nn_ids", [])[:m]
        if same_parent_only:
            pid = a.meta.get("parent_id")
            nn_ids = [nid for nid in nn_ids if pid is None or pid == pid]  # keep as-is; parent filter later
        # Exclude duplicates & already present ids
        for nid in nn_ids:
            if nid not in expanded and nid not in want_ids:
                want_ids.append(nid)

    if not want_ids:
        return list(expanded.values())

    # Try to fetch by id (Haystack v2 stores differ; support both common ways)
    neighbors = []
    try:
        # Preferred, if available
        neighbors = document_store.get_documents_by_id(ids=want_ids)
    except Exception:
        # Fallback via filter
        try:
            neighbors = document_store.filter_documents(filters={"id": {"$in": want_ids}})
        except Exception:
            neighbors = []

    # Optional: enforce same-parent at fetch time
    for n in neighbors:
        if same_parent_only:
            a0 = anchors[0]
            pid_anchor = a0.meta.get("parent_id")
            if pid_anchor is not None and n.meta.get("parent_id") != pid_anchor:
                continue
        expanded.setdefault(n.id, n)

    # Return anchors first, then added neighbors (stable order)
    ordered = anchors + [d for d_id, d in expanded.items() if d_id not in {a.id for a in anchors}]
    return ordered
