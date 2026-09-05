"""A fast, resumable reimplementation of PTS token search.

Upstream `pts` owns the semantics (see ``docs/pts_semantics.md``, pinned at
commit 8334808); this package owns the *control flow*, so that rollouts from
many queries can be batched together and a run can survive a killed session.
"""
