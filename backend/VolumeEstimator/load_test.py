import os
import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(__file__)

def make_request(url, image_path):
    start_time = time.time()
    try:
        with open(image_path, "rb") as f:
            files = {"img": f}
            response = requests.post(url, files=files)
            response.raise_for_status()
            
            # Ensure we actually get a response body
            _ = response.json()
            
        latency = time.time() - start_time
        return True, latency, None
    except Exception as e:
        latency = time.time() - start_time
        return False, latency, str(e)

def calculate_percentile(data, percentile):
    if not data:
        return 0.0
    size = len(data)
    index = (size - 1) * percentile / 100.0
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    
    if upper >= size:
        return data[-1]
    return data[lower] * (1.0 - weight) + data[upper] * weight

def run_load_test(url, image_path, total_requests, concurrency):
    print(f"Starting load test on {url}")
    print(f"Total Requests: {total_requests}, Concurrency: {concurrency}")
    print(f"Image used: {image_path}\n")
    
    latencies = []
    success_count = 0
    failure_count = 0
    errors = {}

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        futures = [
            executor.submit(make_request, url, image_path)
            for _ in range(total_requests)
        ]
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures)):
            success, latency, error_msg = future.result()
            latencies.append(latency)
            
            if success:
                success_count += 1
            else:
                failure_count += 1
                errors[error_msg] = errors.get(error_msg, 0) + 1
            
            # Print progress
            if (i + 1) % max(1, (total_requests // 10)) == 0:
                print(f"Progress: {i + 1}/{total_requests} requests completed...")

    total_time = time.time() - start_time
    
    # Calculate metrics
    if latencies:
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = calculate_percentile(latencies, 50)
        p90_latency = calculate_percentile(latencies, 90)
        p95_latency = calculate_percentile(latencies, 95)
        p99_latency = calculate_percentile(latencies, 99)
    else:
        avg_latency = p50_latency = p90_latency = p95_latency = p99_latency = 0.0

    throughput = success_count / total_time if total_time > 0 else 0

    print("\n" + "="*40)
    print("LOAD TEST RESULTS (For Resume Metrics)")
    print("="*40)
    print(f"Total Time:          {total_time:.2f} seconds")
    print(f"Total Requests:      {total_requests}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests:     {failure_count}")
    print(f"Throughput:          {throughput:.2f} requests/second")
    print("-" * 40)
    print("Latency Metrics (seconds):")
    print(f"  Average:           {avg_latency:.4f}")
    print(f"  P50 (Median):      {p50_latency:.4f}")
    print(f"  P90:               {p90_latency:.4f}")
    print(f"  P95:               {p95_latency:.4f}")
    print(f"  P99:               {p99_latency:.4f}")
    
    if failure_count > 0:
        print("-" * 40)
        print("Error Breakdown:")
        for err, count in errors.items():
            print(f"  - {err}: {count} occurrences")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test VolumeEstimator API")
    parser.add_argument("--url", default="http://0.0.0.0:8000/predict", help="API Endpoint URL")
    parser.add_argument("--image", default=os.path.join(BASE_DIR, "food_volume_estimation/uploads/test2.png"), help="Path to image file")
    parser.add_argument("-n", "--requests", type=int, default=10, help="Total number of requests to make")
    parser.add_argument("-c", "--concurrency", type=int, default=1, help="Number of concurrent requests")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        exit(1)
        
    run_load_test(args.url, args.image, args.requests, args.concurrency)
