#!/usr/bin/env python3
"""
Fix all import issues in Kanibus nodes
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match the problematic import section
        pattern = r'# Import our core system\nimport sys\nimport os\nsys\.path\.append\(os\.path\.dirname\(os\.path\.dirname\(__file__\)\)\)\nfrom src\.(.*)'
        
        def replacement(match):
            module = match.group(1)
            return f"""# Import our core system
try:
    from ..src.{module}
except ImportError:
    # Fallback for development/testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.{module}"""
        
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ Fixed imports in {filepath}")
            return True
        else:
            print(f"⚠️  No import fixes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False

def main():
    """Fix all import issues"""
    nodes_dir = Path(__file__).parent / "nodes"
    
    python_files = list(nodes_dir.glob("*.py"))
    fixed_count = 0
    
    print(f"🔧 Fixing imports in {len(python_files)} files...")
    
    for filepath in python_files:
        if fix_imports_in_file(filepath):
            fixed_count += 1
    
    print(f"\n✅ Fixed imports in {fixed_count}/{len(python_files)} files")

if __name__ == "__main__":
    main()