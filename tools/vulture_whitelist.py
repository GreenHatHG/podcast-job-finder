"""Whitelist for code usage that Vulture cannot infer statically."""

from podcast_job_finder.xiaoyuzhou.episode_parser import _HTMLTextExtractor


VULTURE_WHITELIST_REFERENCES = (
    _HTMLTextExtractor.handle_data,
    _HTMLTextExtractor.handle_starttag,
    _HTMLTextExtractor.handle_endtag,
)
