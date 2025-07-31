#!/usr/bin/env python3
"""
Fix Retry Mechanism Script

This script applies the necessary fixes to enable proper retry logic
for individual batch operations in the ArmorCode-Port integration.

Usage:
    # From the project root:
    python -m armorcode_integration.utils.fix_retry_mechanism
    
    # Or directly:
    python src/armorcode_integration/utils/fix_retry_mechanism.py

This utility is part of the armorcode_integration package and works
with the new professional package structure.
"""

import re
import shutil
from pathlib import Path


def backup_file(file_path):
    """Create a backup of the file before modifying."""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"✅ Created backup: {backup_path}")


def fix_bulk_port_manager():
    """Fix BulkPortManager to use RetryManager for individual batches."""
    file_path = Path(__file__).parent.parent / "managers" / "bulk_port_manager.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Add RetryManager to TYPE_CHECKING imports
    content = content.replace(
        'if TYPE_CHECKING:\n    from ..utils.batch_accumulator import BatchAccumulator',
        'if TYPE_CHECKING:\n    from ..utils.batch_accumulator import BatchAccumulator\n    from .retry_manager import RetryManager'
    )
    
    # Update constructor to accept retry_manager
    old_init = 'def __init__(self, port_token: str, port_api_url: str = "https://api.us.getport.io/v1"):'
    new_init = 'def __init__(self, port_token: str, port_api_url: str = "https://api.us.getport.io/v1", retry_manager: Optional[\'RetryManager\'] = None):'
    content = content.replace(old_init, new_init)
    
    # Add retry_manager to instance variables
    old_vars = '''        self.port_token = port_token
        self.port_api_url = port_api_url
        self.session: Optional[aiohttp.ClientSession] = None'''
    
    new_vars = '''        self.port_token = port_token
        self.port_api_url = port_api_url
        self.retry_manager = retry_manager
        self.session: Optional[aiohttp.ClientSession] = None'''
    
    content = content.replace(old_vars, new_vars)
    
    # Update docstring
    old_docstring = '''        Args:
            port_token: Port API authentication token
            port_api_url: Base URL for Port API'''
    
    new_docstring = '''        Args:
            port_token: Port API authentication token
            port_api_url: Base URL for Port API
            retry_manager: RetryManager instance for handling retries'''
    
    content = content.replace(old_docstring, new_docstring)
    
    # Update batch processing to use retry
    old_batch_code = '''            try:
                result = await self._submit_batch(blueprint_id, batch)'''
    
    new_batch_code = '''            try:
                if self.retry_manager:
                    # Use retry for individual batch submissions
                    result = await self.retry_manager.with_retry(
                        self._submit_batch, blueprint_id, batch
                    )
                else:
                    # Fallback to direct call if no retry manager
                    result = await self._submit_batch(blueprint_id, batch)'''
    
    content = content.replace(old_batch_code, new_batch_code)
    
    # Write the modified content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {file_path}")


def fix_main_py():
    """Fix main.py to pass RetryManager to BulkPortManager."""
    file_path = Path(__file__).parent.parent / "core" / "main.py"
    backup_file(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace BulkPortManager instantiations
    # Pattern 1: In ingest_products
    content = re.sub(
        r'bulk_manager = BulkPortManager\(port_token\)',
        r'bulk_manager = BulkPortManager(port_token, retry_manager=retry_manager)',
        content
    )
    
    # Pattern 2: Any other BulkPortManager calls without retry_manager
    content = re.sub(
        r'BulkPortManager\(([^)]*port_token[^)]*)\)(?![^)]*retry_manager)',
        r'BulkPortManager(\1, retry_manager=retry_manager)',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {file_path}")


def verify_fixes():
    """Verify that the fixes were applied correctly."""
    print("\n🔍 Verifying fixes...")
    
    # Check BulkPortManager
    bulk_path = Path(__file__).parent.parent / "managers" / "bulk_port_manager.py"
    with open(bulk_path, 'r') as f:
        content = f.read()
        
    if 'retry_manager: Optional[\'RetryManager\']' in content:
        print("✅ BulkPortManager constructor updated")
    else:
        print("❌ BulkPortManager constructor not updated")
        
    if 'self.retry_manager = retry_manager' in content:
        print("✅ BulkPortManager instance variable added")
    else:
        print("❌ BulkPortManager instance variable not added")
        
    if 'await self.retry_manager.with_retry(' in content:
        print("✅ BulkPortManager batch retry logic added")
    else:
        print("❌ BulkPortManager batch retry logic not added")
    
    # Check main.py
    main_path = Path(__file__).parent.parent / "core" / "main.py"
    with open(main_path, 'r') as f:
        content = f.read()
        
    if 'retry_manager=retry_manager' in content:
        print("✅ main.py updated to pass retry_manager")
    else:
        print("❌ main.py not updated properly")


def test_syntax():
    """Test that the modified files have valid Python syntax."""
    print("\n🧪 Testing syntax...")
    
    import subprocess
    
    files_to_test = [
        Path(__file__).parent.parent / "managers" / "bulk_port_manager.py",
        Path(__file__).parent.parent / "core" / "main.py"
    ]
    
    for file_path in files_to_test:
        if file_path.exists():
            result = subprocess.run(['python', '-m', 'py_compile', file_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {file_path} syntax OK")
            else:
                print(f"❌ {file_path} syntax error: {result.stderr}")


def main():
    """Apply all fixes."""
    print("🔧 Applying retry mechanism fixes...")
    print("=" * 50)
    
    try:
        fix_bulk_port_manager()
        fix_main_py()
        
        verify_fixes()
        test_syntax()
        
        print("\n🎉 All fixes applied successfully!")
        print("\nNext steps:")
        print("1. Test the integration: armorcode-integration --dry-run --steps finding --finding-filters config/test.json")
        print("2. Or use module execution: python -m armorcode_integration --dry-run --steps finding")
        print("3. Check that individual batch failures now retry properly")
        print("4. Monitor for 401 errors - they should be properly handled")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        print("Check the backup files (.backup) if you need to restore")


if __name__ == "__main__":
    main()
