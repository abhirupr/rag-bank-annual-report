from __future__ import annotations
from typing import Any, Iterator

from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    DocChunk,
    DocMeta,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    DocItem,
    InlineGroup,
    LevelNumber,
    ListGroup,
    SectionHeaderItem,
    TitleItem,
)

class CustomHierarchicalChunker(HierarchicalChunker):
    """
    Custom hierarchical chunker that keeps all the headers at the same level.
    """

    def chunk(
        self,
        dl_doc: DLDocument,
        **kwargs: Any,
    ) -> Iterator[DocChunk]:
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        # level -> list of headers seen (siblings) under the current parent
        headings_by_level: dict[LevelNumber, list[str]] = {}
        # level -> sealed flag; after content appears, the level’s siblings reset on next header at that level
        sealed_by_level: dict[LevelNumber, bool] = {}

        visited: set[str] = set()
        ser_res = create_ser_result()
        excluded_refs = my_doc_ser.get_excluded_refs(**kwargs)

        for item, _level in dl_doc.iterate_items(with_groups=True):
            if item.self_ref in excluded_refs:
                continue

            if isinstance(item, (TitleItem, SectionHeaderItem)):
                lvl: LevelNumber = item.level if isinstance(item, SectionHeaderItem) else 0

                # remove deeper levels (scope reset)
                to_del = [k for k in headings_by_level if k > lvl]
                for k in to_del:
                    headings_by_level.pop(k, None)
                    sealed_by_level.pop(k, None)

                # if content was already emitted under this level, start a fresh sibling list
                if sealed_by_level.get(lvl, False):
                    headings_by_level[lvl] = []
                    sealed_by_level[lvl] = False

                # append this header to the sibling list at this level
                lst = headings_by_level.get(lvl, [])
                if not lst or lst[-1] != item.text:
                    lst = lst + [item.text] if lst else [item.text]
                    headings_by_level[lvl] = lst
                continue

            elif (
                isinstance(item, (ListGroup, InlineGroup, DocItem))
                and item.self_ref not in visited
            ):
                ser_res = my_doc_ser.serialize(item=item, visited=visited)
            else:
                continue

            if not ser_res.text:
                continue

            doc_items = [u.item for u in ser_res.spans] if ser_res.spans else []
            if not doc_items:
                continue

            # flatten all headers across levels (ascending by level)
            all_headings: list[str] = []
            for k in sorted(headings_by_level):
                all_headings.extend(headings_by_level[k])

            yield DocChunk(
                text=ser_res.text,
                meta=DocMeta(
                    doc_items=doc_items,
                    headings=all_headings or None,
                    origin=dl_doc.origin,
                ),
            )

            # seal all current levels after emitting content,
            # so the next same-level header starts a fresh sibling list
            for k in list(headings_by_level.keys()):
                sealed_by_level[k] = True


class CustomHybridChunker(HybridChunker):
    """
    Hybrid chunker that uses CustomHierarchicalChunker internally.
    """

    def model_post_init(self, __context: Any) -> None:
        parent = getattr(super(), "model_post_init", None)
        if callable(parent):
            parent(__context)
        self._inner_chunker = CustomHierarchicalChunker(
            serializer_provider=self.serializer_provider
        )