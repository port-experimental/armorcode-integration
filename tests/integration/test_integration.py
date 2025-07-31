#!/usr/bin/env python3
"""
Integration Test for DirectArmorCodeClient

This script tests the complete integration of the DirectArmorCodeClient
with the existing Port ingestion logic to ensure everything works correctly.
"""

import asyncio
import os
import logging
from dotenv import load_dotenv
from armorcode_client import ArmorCodeClient, ArmorCodeAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_integration():
    """Test the complete integration flow."""
    print("🚀 Starting DirectArmorCodeClient Integration Test...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("ARMORCODE_API_KEY")
    
    if not api_key:
        print("❌ ERROR: ARMORCODE_API_KEY not found in environment")
        return False
        
    try:
        # Test 1: Client creation and connection
        print("\n📡 Test 1: Testing client creation and connection...")
        async with ArmorCodeClient(api_key) as ac_client:
            
            # Test connection
            if await ac_client.test_connection():
                print("✅ Connection test successful")
            else:
                print("❌ Connection test failed")
                return False
                
            # Test 2: Single page retrieval
            print("\n📄 Test 2: Testing single page retrieval...")
            response = await ac_client.get_findings_page(size=5)
            
            if response.get('success'):
                data = response.get('data', {})
                findings = data.get('findings', [])
                print(f"✅ Retrieved {len(findings)} findings in single page")
                
                # Validate data structure
                if findings and len(findings) > 0:
                    sample_finding = findings[0]
                    required_fields = ['id', 'title', 'status', 'severity', 'source', 'subProduct']
                    
                    missing_fields = [field for field in required_fields if field not in sample_finding]
                    if missing_fields:
                        print(f"❌ Missing required fields in finding: {missing_fields}")
                        return False
                    else:
                        print("✅ Finding data structure validation passed")
                        
                        # Print sample data
                        print(f"   Sample finding: ID={sample_finding['id']}, "
                              f"Title='{sample_finding['title'][:50]}...', "
                              f"Status={sample_finding['status']}, "
                              f"Severity={sample_finding['severity']}")
                else:
                    print("⚠️  No findings returned in sample")
            else:
                print("❌ Single page retrieval failed")
                return False
                
            # Test 3: Limited pagination test (first 2 pages only)
            print("\n📚 Test 3: Testing pagination (limited test)...")
            all_findings = []
            page_count = 0
            after_key = None
            
            # Test only first 2 pages to avoid long-running test
            for page_num in range(1, 3):
                page_response = await ac_client.get_findings_page(size=100, after_key=after_key)
                
                if page_response.get('success'):
                    page_data = page_response.get('data', {})
                    page_findings = page_data.get('findings', [])
                    
                    if page_findings:
                        all_findings.extend(page_findings)
                        after_key = page_data.get('afterKey')
                        page_count += 1
                        print(f"✅ Page {page_num}: Retrieved {len(page_findings)} findings "
                              f"(total: {len(all_findings)})")
                        
                        if after_key is None:
                            print("   No more pages available")
                            break
                    else:
                        print(f"   Page {page_num}: No findings returned")
                        break
                else:
                    print(f"❌ Page {page_num} retrieval failed")
                    return False
                    
            print(f"✅ Pagination test completed: {len(all_findings)} findings across {page_count} pages")
            
            # Test 4: Data compatibility with existing processing logic  
            print("\n🔄 Test 4: Testing data compatibility with existing processing logic...")
            
            if all_findings:
                sample_finding = all_findings[0]
                
                # Test the field mappings we use in main.py
                try:
                    # These are the fields we extract in the updated main.py
                    finding_id = sample_finding["id"]
                    title = sample_finding.get("title", "Unknown Finding")
                    status = sample_finding.get("status", "UNKNOWN")
                    severity = sample_finding.get("severity", "UNKNOWN")
                    source = sample_finding.get("source", "Unknown Tool")
                    finding_url = sample_finding.get("findingUrl")
                    sub_product_id = sample_finding["subProduct"]["id"]
                    cves = sample_finding.get("cve", [])
                    
                    print(f"✅ Field extraction successful:")
                    print(f"   ID: {finding_id}")
                    print(f"   Title: '{title[:30]}...'")
                    print(f"   Status: {status}")
                    print(f"   Severity: {severity}")
                    print(f"   Source: {source}")
                    print(f"   URL: {finding_url[:50] if finding_url else 'None'}...")
                    print(f"   SubProduct ID: {sub_product_id}")
                    print(f"   CVEs: {len(cves)} found")
                    
                except KeyError as e:
                    print(f"❌ Missing required field in finding data: {e}")
                    return False
                except Exception as e:
                    print(f"❌ Error processing finding data: {e}")
                    return False
            else:
                print("⚠️  No findings available for compatibility testing")
                
        print("\n🎉 All integration tests passed successfully!")
        return True
        
    except ArmorCodeAPIError as e:
        print(f"❌ ArmorCode API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

async def main():
    """Main test runner."""
    success = await test_integration()
    
    if success:
        print("\n✅ INTEGRATION TEST PASSED - Ready for production use!")
        exit(0)
    else:
        print("\n❌ INTEGRATION TEST FAILED - Please check the issues above")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 