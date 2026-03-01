"""
Streamlit app for visualizing TAV document structure and search.
Run: streamlit run app.py
"""

import os
import time
import tempfile
import streamlit as st

st.set_page_config(page_title="TAV Explorer", page_icon="", layout="wide")

# Lazy imports to avoid slow startup when just loading the page
@st.cache_resource
def load_backend(model_name):
    from tav.embedder import get_backend
    return get_backend(model_name)


def parse_uploaded_pdf(uploaded_file):
    from tav.structural_parser import parse_pdf
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    tree = parse_pdf(tmp_path)
    os.unlink(tmp_path)
    return tree


def tree_to_display(nodes, depth=0):
    """Recursively build displayable tree items."""
    items = []
    for node in nodes:
        items.append({
            "indent": depth,
            "id": node.node_id,
            "level": node.level,
            "title": node.title,
            "pages": f"pp. {node.page_start}-{node.page_end}",
            "text_len": len(node.text),
            "text_preview": node.text[:300].replace("\n", " ") if node.text else "",
            "num_children": len(node.children),
        })
        items.extend(tree_to_display(node.children, depth + 1))
    return items


def render_tree_card(item):
    indent_px = item["indent"] * 28
    level_colors = {1: "#6366f1", 2: "#8b5cf6", 3: "#a78bfa"}
    color = level_colors.get(item["level"], "#c4b5fd")
    level_labels = {1: "Chapter", 2: "Section", 3: "Paragraph"}
    label = level_labels.get(item["level"], f"L{item['level']}")

    st.markdown(f"""
    <div style="margin-left: {indent_px}px; padding: 8px 12px; margin-bottom: 4px;
                border-left: 3px solid {color}; background: rgba(99,102,241,0.05);
                border-radius: 0 6px 6px 0;">
        <span style="color: {color}; font-size: 11px; font-weight: 600;">{label}</span>
        <span style="color: #94a3b8; font-size: 11px; margin-left: 8px;">[{item['id']}]</span>
        <span style="color: #94a3b8; font-size: 11px; margin-left: 8px;">{item['pages']}</span>
        <br/>
        <span style="font-size: 14px; font-weight: 500;">{item['title']}</span>
        <span style="color: #64748b; font-size: 11px; margin-left: 8px;">{item['text_len']} chars</span>
    </div>
    """, unsafe_allow_html=True)


def render_search_result(r, rank):
    score_pct = min(100, int(r["score"] * 100))
    bar_color = "#22c55e" if score_pct > 60 else "#eab308" if score_pct > 40 else "#ef4444"
    st.markdown(f"""
    <div style="padding: 12px 16px; margin-bottom: 8px; background: rgba(99,102,241,0.06);
                border-radius: 8px; border: 1px solid rgba(99,102,241,0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-weight: 600; font-size: 15px;">#{rank} {r['hierarchy_path']}</span>
            <span style="color: #94a3b8; font-size: 12px;">pp. {r['page_start']}-{r['page_end']}</span>
        </div>
        <div style="margin-top: 6px; background: #1e293b; border-radius: 4px; height: 6px; overflow: hidden;">
            <div style="background: {bar_color}; height: 100%; width: {score_pct}%;"></div>
        </div>
        <span style="color: #94a3b8; font-size: 11px;">Score: {r['score']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)


# --- UI ---

st.markdown("""
<h1 style="margin-bottom: 0;">TAV Explorer</h1>
<p style="color: #94a3b8; margin-top: 4px;">Topology-Aware Vector Routing — upload a PDF, see the structure, search it.</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    with st.spinner("Parsing PDF structure..."):
        if "tree" not in st.session_state or st.session_state.get("filename") != uploaded.name:
            tree = parse_uploaded_pdf(uploaded)
            st.session_state.tree = tree
            st.session_state.filename = uploaded.name
            st.session_state.indexed = False

    tree = st.session_state.tree
    tree_items = tree_to_display(tree)

    from tav.structural_parser import get_all_nodes_flat
    all_nodes = get_all_nodes_flat(tree)
    chapters = [n for n in all_nodes if n.level == 1]
    sections = [n for n in all_nodes if n.level == 2]
    paras = [n for n in all_nodes if n.level >= 3]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", len(all_nodes))
    col2.metric("Chapters", len(chapters))
    col3.metric("Sections", len(sections))
    col4.metric("Paragraphs", len(paras) if paras else len(sections))

    tab_tree, tab_search = st.tabs(["Document Tree", "Search"])

    with tab_tree:
        st.markdown("### Document Hierarchy")
        for item in tree_items:
            render_tree_card(item)
            if item["text_preview"]:
                indent_px = (item["indent"] + 1) * 28
                with st.expander(f"Preview text ({item['text_len']} chars)", expanded=False):
                    st.text(item["text_preview"] + ("..." if item["text_len"] > 300 else ""))

    with tab_search:
        st.markdown("### Semantic Zoom Search")

        embed_model = st.selectbox("Embedding model", ["all-MiniLM-L6-v2", "openai"], index=0)

        if not st.session_state.get("indexed"):
            if st.button("Build Index", type="primary"):
                with st.spinner("Building FAISS index..."):
                    from tav.embedder import build_index
                    idx = build_index(tree, embed_model=embed_model)
                    st.session_state.index_data = idx
                    st.session_state.indexed = True
                    st.session_state.embed_model = embed_model
                st.success(f"Indexed! {idx['config']['num_chapters']} chapters, "
                           f"{idx['config']['num_sections']} sections, "
                           f"{idx['config']['num_paragraphs']} paragraphs")
                st.rerun()
        else:
            st.success("Index ready")
            query = st.text_input("Enter your query")

            c1, c2, c3 = st.columns(3)
            k_chap = c1.slider("K chapters", 1, 10, 3)
            k_sec = c2.slider("K sections", 1, 20, 5)
            k_para = c3.slider("K paragraphs", 1, 30, 10)

            if query:
                from tav.search import semantic_zoom_search
                from tav.context_retriever import retrieve_context

                t0 = time.time()
                results = semantic_zoom_search(
                    query, st.session_state.index_data,
                    embed_model=st.session_state.embed_model,
                    k_chapters=k_chap, k_sections=k_sec, k_paragraphs=k_para,
                )
                elapsed = time.time() - t0

                st.markdown(f"**Found {len(results)} results in {elapsed*1000:.1f}ms**")

                for i, r in enumerate(results):
                    render_search_result(r, i + 1)

                ctx = retrieve_context(
                    results,
                    st.session_state.index_data["paragraph_meta"],
                    st.session_state.index_data["section_meta"],
                    st.session_state.index_data["chapter_meta"],
                )

                with st.expander("Assembled Context", expanded=False):
                    st.markdown(f"*{ctx['token_count']} tokens*")
                    st.text(ctx["context"])

else:
    st.info("Upload a PDF to get started.")
