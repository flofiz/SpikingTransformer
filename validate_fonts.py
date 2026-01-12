#!/usr/bin/env python3
"""
Standalone Font Validation Script

This script scans font directories and validates each font for OCR compatibility.
Results are cached to `valid_fonts.json` for fast loading during training.

Usage:
    python validate_fonts.py                    # Scan and validate all fonts
    python validate_fonts.py --output fonts.json  # Custom output file
    python validate_fonts.py --workers 16       # Use 16 parallel workers
    python validate_fonts.py --quick            # Skip validation, just list .ttf/.otf files
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

# Test characters for font validation (Latin alphabet + accents + numbers)
FONT_TEST_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789éèêëàâäùûüôöîïçÉÈÊËÀÂÄÙÛÜÔÖÎÏÇ"


def validate_single_font(font_path: str, test_chars: str = FONT_TEST_CHARS, test_size: int = 28) -> Tuple[str, bool, str]:
    """
    Validates a single font file.
    
    Args:
        font_path: Path to the font file
        test_chars: Characters to test rendering
        test_size: Font size for testing
        
    Returns:
        Tuple of (font_path, is_valid, error_message)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        font = ImageFont.truetype(font_path, size=test_size)
        
        # Create test image
        test_img = Image.new('L', (400, 80), color=255)
        draw = ImageDraw.Draw(test_img)
        
        # Test bounding box
        try:
            bbox = font.getbbox(test_chars[:30])
            if bbox and (bbox[2] - bbox[0]) > 0 and (bbox[3] - bbox[1]) > 0:
                # Additional test: try to render
                draw.text((5, 5), "Test ABC 123 éàù", fill=0, font=font)
                return (font_path, True, "")
            else:
                return (font_path, False, "Empty bounding box")
        except Exception as e:
            # Try drawing anyway
            draw.text((5, 5), "Test ABC 123", fill=0, font=font)
            return (font_path, True, "")
            
    except Exception as e:
        return (font_path, False, str(e))


def scan_font_files(font_dirs: List[str]) -> List[str]:
    """
    Scans directories for font files without validation.
    
    Args:
        font_dirs: List of directories to scan
        
    Returns:
        List of font file paths
    """
    font_files = []
    
    for font_dir in font_dirs:
        if not os.path.exists(font_dir):
            print(f"Warning: Directory not found: {font_dir}")
            continue
            
        for root, dirs, files in os.walk(font_dir):
            for file in files:
                ext = file.lower()
                if ext.endswith('.ttf') or ext.endswith('.otf'):
                    font_files.append(os.path.join(root, file))
    
    return font_files


def validate_fonts_parallel(
    font_files: List[str],
    num_workers: int = 8,
    test_chars: str = FONT_TEST_CHARS,
    show_progress: bool = True
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Validates fonts in parallel using ThreadPoolExecutor.
    
    Args:
        font_files: List of font file paths to validate
        num_workers: Number of parallel workers
        test_chars: Characters to test rendering
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (valid_fonts, failed_fonts with errors)
    """
    valid_fonts = []
    failed_fonts = []
    total = len(font_files)
    processed = 0
    lock = Lock()
    start_time = time.time()
    
    def update_progress():
        nonlocal processed
        with lock:
            processed += 1
            if show_progress and processed % 100 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                print(f"\r  Progress: {processed}/{total} ({processed*100//total}%) - "
                      f"{rate:.1f} fonts/s - ETA: {eta:.0f}s", end="", flush=True)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(validate_single_font, path, test_chars): path 
            for path in font_files
        }
        
        for future in as_completed(futures):
            path, is_valid, error = future.result()
            
            if is_valid:
                with lock:
                    valid_fonts.append(path)
            else:
                with lock:
                    failed_fonts.append((path, error))
            
            update_progress()
    
    if show_progress:
        print()  # New line after progress
    
    return valid_fonts, failed_fonts


def main():
    parser = argparse.ArgumentParser(
        description="Validate fonts for OCR compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output", "-o",
        default="valid_fonts.json",
        help="Output JSON file for valid fonts (default: valid_fonts.json)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: skip validation, just list font files"
    )
    parser.add_argument(
        "--font-dirs",
        nargs="+",
        help="Custom font directories to scan (default: Fonts, fonts_HW)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including failed fonts"
    )
    
    args = parser.parse_args()
    
    # Default font directories
    base_path = Path(__file__).parent
    if args.font_dirs:
        font_dirs = args.font_dirs
    else:
        font_dirs = [
            str(base_path / "Fonts"),
            str(base_path / "fonts_HW"),
        ]
    
    print("=" * 60)
    print("Font Validation Tool")
    print("=" * 60)
    print(f"\nScanning directories:")
    for d in font_dirs:
        exists = "✓" if os.path.exists(d) else "✗"
        print(f"  {exists} {d}")
    
    # Scan for font files
    print("\nScanning for font files...")
    font_files = scan_font_files(font_dirs)
    print(f"  Found {len(font_files)} font files")
    
    if not font_files:
        print("\nNo fonts found. Exiting.")
        return 1
    
    if args.quick:
        # Quick mode: just save the file list without validation
        print("\nQuick mode: Skipping validation")
        valid_fonts = font_files
        failed_fonts = []
    else:
        # Validate fonts in parallel
        print(f"\nValidating fonts (using {args.workers} workers)...")
        valid_fonts, failed_fonts = validate_fonts_parallel(
            font_files,
            num_workers=args.workers,
            show_progress=True
        )
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Total scanned: {len(font_files)}")
    print(f"  Valid fonts:   {len(valid_fonts)} ({len(valid_fonts)*100//len(font_files)}%)")
    print(f"  Failed fonts:  {len(failed_fonts)}")
    
    if args.verbose and failed_fonts:
        print(f"\nFailed fonts:")
        for path, error in failed_fonts[:20]:  # Show first 20
            print(f"  - {os.path.basename(path)}: {error[:50]}")
        if len(failed_fonts) > 20:
            print(f"  ... and {len(failed_fonts) - 20} more")
    
    # Save to JSON
    output_path = args.output
    output_data = {
        "version": "1.0",
        "font_dirs": font_dirs,
        "validated": not args.quick,
        "total_scanned": len(font_files),
        "valid_count": len(valid_fonts),
        "fonts": valid_fonts
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(valid_fonts)} fonts to: {output_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
