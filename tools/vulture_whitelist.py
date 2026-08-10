"""Whitelist for dynamic code usage that Vulture cannot infer statically."""

# pylint: disable=pointless-statement

from podcast_job_finder.episode.parser import _HTMLTextExtractor


class _DynamicAttributeReferences:
    def __getattr__(self, _name: str) -> object:
        return object()


_dynamic_attribute = _DynamicAttributeReferences()

# HTMLParser dispatches these callbacks by method name.
_HTMLTextExtractor.handle_data
_HTMLTextExtractor.handle_starttag
_HTMLTextExtractor.handle_endtag

# PyTorch invokes forward through Module.__call__.
_dynamic_attribute.forward

# External libraries read these configured attributes internally.
_dynamic_attribute.graph_optimization_level
_dynamic_attribute.intra_op_num_threads
_dynamic_attribute.dither
_dynamic_attribute.snip_edges
_dynamic_attribute.num_bins
_dynamic_attribute.debug_mel
_dynamic_attribute.pooler
