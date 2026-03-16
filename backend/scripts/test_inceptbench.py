#!/usr/bin/env python3
"""
Test script for InceptBench integration.

Usage:
    # Test the simulator directly
    python scripts/test_inceptbench.py --direct

    # Test via API (requires server running)
    python scripts/test_inceptbench.py --api

    # Test both
    python scripts/test_inceptbench.py --all
"""

import argparse
import asyncio
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


SAMPLE_QUESTION = {
    "type": "question",
    "text": "What is the primary function of the Electoral College in the United States?",
    "options": [
        "To directly elect members of Congress",
        "To formally elect the President and Vice President",
        "To approve Supreme Court nominations",
        "To ratify constitutional amendments"
    ],
    "correct_answer": "B",
    "grade": "11",
    "subject": "AP Government",
}

SAMPLE_MATH_QUESTION = {
    "type": "question",
    "text": "A rectangle has a length of 12 cm and a width of 5 cm. What is its area?",
    "options": [
        "17 cm²",
        "34 cm²",
        "60 cm²",
        "120 cm²"
    ],
    "correct_answer": "C",
    "grade": "6",
    "subject": "Mathematics",
}


async def test_direct():
    """Test the InceptBenchSimulator directly."""
    print("\n" + "=" * 60)
    print("DIRECT SIMULATOR TEST")
    print("=" * 60)

    from app.services.inceptbench_simulator import InceptBenchSimulator

    simulator = InceptBenchSimulator()

    def progress(cur, total, msg):
        print(f"  [{cur}/{total}] {msg}")

    print("\n--- Test 1: AP Government Question ---")
    result = await simulator.simulate_content(
        content=SAMPLE_QUESTION,
        population_config={"size": 15},
        progress_callback=progress,
    )

    print("\nResults:")
    print(f"  Accuracy: {result.accuracy:.1%}")
    print(f"  Difficulty (IRT): {result.difficulty_irt:.2f}")
    print(f"  Discrimination (IRT): {result.discrimination_irt:.2f}")
    print(f"  Avg Time: {result.avg_time_seconds:.1f}s")
    print(f"  Engagement: {result.engagement_score:.1%}")

    print("\n  By Archetype:")
    for arch, perf in result.by_archetype.items():
        print(f"    {arch}: {perf.accuracy:.0%} accuracy, {perf.count} students")

    if result.concerns:
        print("\n  Concerns:")
        for c in result.concerns:
            print(f"    - {c}")

    print("\n--- Test 2: Math Question ---")
    result2 = await simulator.simulate_content(
        content=SAMPLE_MATH_QUESTION,
        population_config={"size": 10},
        progress_callback=progress,
    )

    print("\nResults:")
    print(f"  Accuracy: {result2.accuracy:.1%}")
    print(f"  Difficulty (IRT): {result2.difficulty_irt:.2f}")

    print("\n✅ Direct tests passed!")
    return True


def test_api():
    """Test the API endpoints."""
    print("\n" + "=" * 60)
    print("API ENDPOINT TEST")
    print("=" * 60)

    try:
        import requests
    except ImportError:
        print("❌ requests library not installed. Run: pip install requests")
        return False

    base_url = os.environ.get("MIROFISH_API_URL", "http://localhost:5001")

    # Health check
    print(f"\n--- Health Check ({base_url}/api/inceptbench/health) ---")
    try:
        resp = requests.get(f"{base_url}/api/inceptbench/health", timeout=5)
        if resp.status_code == 200:
            print(f"  Status: OK")
            print(f"  Response: {json.dumps(resp.json(), indent=2)}")
        else:
            print(f"  ❌ Status: {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to {base_url}")
        print("  Make sure the MiroFish server is running (npm run backend)")
        return False

    # Simulate endpoint
    print(f"\n--- Simulate Endpoint ({base_url}/api/inceptbench/simulate) ---")
    resp = requests.post(
        f"{base_url}/api/inceptbench/simulate",
        json={
            "content": SAMPLE_QUESTION,
            "population": {"size": 10},
            "include_responses": True,
        },
        timeout=60,
    )

    if resp.status_code == 200:
        data = resp.json()
        if data.get("success"):
            result = data["data"]
            print(f"  Accuracy: {result['aggregate']['accuracy']:.1%}")
            print(f"  Difficulty: {result['aggregate']['difficulty_irt']:.2f}")
            print(f"  Simulation time: {result['simulation_time_ms']}ms")
            print(f"  Responses: {len(result.get('responses', []))} students")
        else:
            print(f"  ❌ Error: {data.get('error')}")
            return False
    else:
        print(f"  ❌ Status: {resp.status_code}")
        print(f"  Response: {resp.text}")
        return False

    # Archetypes endpoint
    print(f"\n--- Archetypes Endpoint ({base_url}/api/inceptbench/archetypes) ---")
    resp = requests.get(f"{base_url}/api/inceptbench/archetypes", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        archetypes = data["data"]["archetypes"]
        print(f"  Available archetypes: {len(archetypes)}")
        for name in archetypes:
            print(f"    - {name}")
    else:
        print(f"  ❌ Status: {resp.status_code}")
        return False

    print("\n✅ API tests passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test InceptBench integration")
    parser.add_argument("--direct", action="store_true", help="Test simulator directly")
    parser.add_argument("--api", action="store_true", help="Test API endpoints")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    if not any([args.direct, args.api, args.all]):
        args.all = True  # Default to all

    success = True

    if args.direct or args.all:
        success = asyncio.run(test_direct()) and success

    if args.api or args.all:
        success = test_api() and success

    print("\n" + "=" * 60)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
