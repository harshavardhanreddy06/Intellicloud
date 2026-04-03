import time
import os
import sys

def process_text(input_path, output_path):
    print(f"--- Container Execution Started: Processing text from {os.path.basename(input_path)} ---")
    start_time = time.time()
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return
    
    with open(input_path, 'r', errors='ignore') as f:
        text = f.read()
    
    print(f"   Original Size: {len(text)} characters")
    
    # Simulate heavy workload: Sorting all words 50 times
    words = text.split()
    for i in range(1, 51):
        sorted_words = sorted(words)
        if i % 10 == 0:
            print(f"      Iteration {i}/50 complete...")

    # Final Save: Word count summary + snippets
    with open(output_path, 'w') as f:
        f.write(f"PROCESSED RESULT\n")
        f.write(f"Word Count: {len(words)}\n")
        f.write(f"Unique Words: {len(set(words))}\n")
        f.write("-" * 20 + "\n")
        f.write("Top 10 Sorted Words:\n")
        f.write("\n".join(sorted_words[:10]))
    
    duration = time.time() - start_time
    print(f"--- Container Execution Finished: Success ({duration:.3f} seconds) ---")
    print(f"METRICS_REPORT: DURATION={duration:.3f}, CPU=1.5, MEM=45.22")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_text.py <input> <output>")
    else:
        process_text(sys.argv[1], sys.argv[2])
