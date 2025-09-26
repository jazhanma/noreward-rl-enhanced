#!/usr/bin/env python3
"""
Script to automatically fix common linting issues in the codebase.
"""
import os
import re
import sys
from pathlib import Path


def fix_whitespace_issues(content):
    """Fix common whitespace issues."""
    # Remove trailing whitespace
    content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Remove blank lines with whitespace
    content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
    
    # Ensure newline at end of file
    if content and not content.endswith('\n'):
        content += '\n'
    
    return content


def fix_import_issues(content):
    """Fix import issues."""
    lines = content.split('\n')
    fixed_lines = []
    imports = []
    other_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
            imports.append(line)
        else:
            other_lines.append(line)
    
    # Add imports at the top
    fixed_lines.extend(imports)
    if imports and other_lines:
        fixed_lines.append('')
    fixed_lines.extend(other_lines)
    
    return '\n'.join(fixed_lines)


def fix_f_strings(content):
    """Fix f-string issues."""
    # Remove f-strings that don't have placeholders
    content = re.sub(r'f"([^"]*)"', r'"\1"', content)
    content = re.sub(r"f'([^']*)'", r"'\1'", content)
    return content


def fix_comparison_issues(content):
    """Fix comparison issues."""
    # Fix '== False' to 'is False'
    content = re.sub(r'== False', 'is False', content)
    content = re.sub(r'== True', 'is True', content)
    return content


def fix_file(file_path):
    """Fix a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_whitespace_issues(content)
        content = fix_f_strings(content)
        content = fix_comparison_issues(content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """Main function."""
    # Get all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")


if __name__ == '__main__':
    main()
