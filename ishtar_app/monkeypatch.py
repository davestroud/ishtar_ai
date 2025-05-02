#!/usr/bin/env python3
"""
Monkeypatch module for fixing Streamlit's module path extraction with torch.

This module patches Streamlit's module path extraction to avoid errors with torch modules.
The error occurs because torch has custom attribute handling that conflicts with Streamlit's
module path extraction logic.
"""

import sys
import types
import logging
import importlib

logger = logging.getLogger(__name__)


def apply_torch_patches():
    """Apply all monkeypatches for torch compatibility with Streamlit"""
    # Fix for torch.__path__._path error
    fix_path_extraction()

    # Fix the asyncio error
    fix_asyncio_loop()

    # Add a fake __path__ attribute to torch._classes
    fix_torch_classes()

    logger.info("Applied monkeypatches for torch compatibility")


def fix_path_extraction():
    """Fix Streamlit's module path extraction for torch modules"""
    try:
        import streamlit as st
        import inspect

        # First try to get the module using its full path (for newer Streamlit versions)
        try:
            # Try to find the local_sources_watcher module
            for module_name in sys.modules:
                if (
                    module_name.endswith("local_sources_watcher")
                    and "streamlit" in module_name
                ):
                    lsw = sys.modules[module_name]
                    break
            else:
                # If not found by direct name, try to find it in streamlit modules
                lsw = None
                for name, module in sys.modules.items():
                    if name.startswith("streamlit") and hasattr(module, "watcher"):
                        try:
                            lsw = module.watcher.local_sources_watcher
                            break
                        except (AttributeError, ImportError):
                            continue

                # If still not found, try direct import
                if lsw is None:
                    try:
                        lsw = importlib.import_module(
                            "streamlit.watcher.local_sources_watcher"
                        )
                    except ImportError:
                        logger.warning(
                            "Could not find streamlit.watcher.local_sources_watcher module"
                        )
                        return
        except Exception as e:
            logger.warning(f"Error finding local_sources_watcher module: {e}")
            return

        # Check if we found the module
        if lsw is None:
            logger.warning("Could not locate Streamlit's local_sources_watcher module")
            return

        # Check which functions we need to patch
        functions_to_patch = {}

        # Try to find extract_paths
        if hasattr(lsw, "extract_paths"):
            functions_to_patch["extract_paths"] = lsw.extract_paths

        # Try to find get_module_paths
        if hasattr(lsw, "get_module_paths"):
            functions_to_patch["get_module_paths"] = lsw.get_module_paths

        # In newer Streamlit, the function might be in a class
        if not functions_to_patch:
            # Look for classes that might contain these functions
            for name, obj in inspect.getmembers(lsw):
                if inspect.isclass(obj):
                    if hasattr(obj, "extract_paths") or hasattr(
                        obj, "get_module_paths"
                    ):
                        logger.info(
                            f"Found class {name} that might contain path extraction functions"
                        )
                        # Future enhancement: patch class methods if needed

            # If we still can't find the functions, try a more generic approach
            logger.info(
                "Could not find specific functions to patch, applying generic fix"
            )
            _apply_generic_streamlit_fix()
            return

        # Define our patched extract_paths function
        def patched_extract_paths(module):
            """Safe version of extract_paths that handles torch modules"""
            if getattr(module, "__name__", "").startswith("torch"):
                return []

            if not hasattr(module, "__path__"):
                return []

            try:
                # Safely try to get the paths from the module
                if isinstance(module.__path__, types.ModuleType):
                    return []

                if hasattr(module.__path__, "_path"):
                    return list(module.__path__._path)
                else:
                    return []
            except (AttributeError, RuntimeError, TypeError):
                return []

        # Define our patched get_module_paths function
        def patched_get_module_paths(module):
            """Safe version of get_module_paths that handles torch modules"""
            try:
                # Skip torch modules entirely
                if getattr(module, "__name__", "").startswith("torch"):
                    return set()

                if "extract_paths" in functions_to_patch:
                    original_get_module_paths = functions_to_patch.get(
                        "get_module_paths"
                    )
                    if original_get_module_paths:
                        return original_get_module_paths(module)
                return set()
            except (RuntimeError, AttributeError):
                return set()

        # Apply the patches
        if "extract_paths" in functions_to_patch:
            lsw.extract_paths = patched_extract_paths
            logger.info("Patched Streamlit's extract_paths function")

        if "get_module_paths" in functions_to_patch:
            lsw.get_module_paths = patched_get_module_paths
            logger.info("Patched Streamlit's get_module_paths function")

    except ImportError:
        logger.warning(
            "Could not patch Streamlit's module path extraction (module not found)"
        )
    except Exception as e:
        logger.error(f"Error patching Streamlit's module path extraction: {e}")
        # Apply a generic fix as fallback
        _apply_generic_streamlit_fix()


def _apply_generic_streamlit_fix():
    """Apply a more generic fix for Streamlit compatibility with torch"""
    try:
        # Make torch.__path__ more robust
        import torch

        # Create a simple list-like object that won't cause errors
        class SafePath(list):
            def __getattr__(self, name):
                return []

        # Add special path attributes
        torch.__path__ = SafePath()

        # If _classes exists, add __path__ to it too
        if hasattr(torch, "_classes"):
            torch._classes.__path__ = SafePath()

        logger.info("Applied generic Streamlit compatibility fix for torch")
    except ImportError:
        logger.warning("torch module not found, skipping generic fix")
    except Exception as e:
        logger.error(f"Error applying generic Streamlit fix: {e}")


def fix_asyncio_loop():
    """Fix the asyncio loop error in Streamlit"""
    try:
        import asyncio

        # Create a new event loop if there isn't one
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info("Created new asyncio event loop")
    except ImportError:
        logger.warning("asyncio module not found")
    except Exception as e:
        logger.error(f"Error fixing asyncio loop: {e}")


def fix_torch_classes():
    """Add a fake __path__ attribute to torch._classes"""
    try:
        import torch
        import types

        # Create a fake module-like object that doesn't error on attribute access
        class SafeModule(types.ModuleType):
            def __getattr__(self, name):
                return None

        # Add fake __path__ attribute to torch.classes
        if hasattr(torch, "_classes"):
            if not hasattr(torch._classes, "__path__"):
                path_obj = SafeModule("__path__")
                path_obj._path = []
                torch._classes.__path__ = path_obj
                logger.info("Added fake __path__ attribute to torch._classes")

        # Also add it to torch itself for good measure
        if not hasattr(torch, "__path__"):
            path_obj = SafeModule("__path__")
            path_obj._path = []
            torch.__path__ = path_obj
            logger.info("Added fake __path__ attribute to torch")
    except ImportError:
        logger.warning("torch module not found")
    except Exception as e:
        logger.error(f"Error fixing torch._classes: {e}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Apply all patches
    apply_torch_patches()

    print("Applied all monkeypatches for torch compatibility with Streamlit")
