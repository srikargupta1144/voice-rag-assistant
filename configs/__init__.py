"""
File: configs/__init__.py

Description:
    Typed settings tree for the whole application. See configs.settings for the order in
    which setting sources take precedence.

Libraries used:
    configs.settings - project; provides REPO, Settings and get_settings for re-export

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header in the standard format.
"""

from configs.settings import REPO, Settings, get_settings

__all__ = ["REPO", "Settings", "get_settings"]
