import subprocess
import re

score_list = []
for i in range(1000):
    # 執行目標 script，捕捉輸出 (請確認 Python 執行環境正確)
    result = subprocess.run(["python", "simple_custom_taxi_env.py"],
                            capture_output=True, text=True)
    output = result.stdout.strip()
    
    # 利用正則表達式擷取 "Final Score: ..." 中的分數
    match = re.search(r'Final Score:\s*(-?\d+\.\d+)', output)
    if match:
        score = float(match.group(1))
        score_list.append(score)
        print(f"Iteration {i}: Final Score: {score}")
    else:
        print(f"Iteration {i}: no output")

# 過濾出小於 -1000 的分數
scores_below_threshold = [s for s in score_list if s < -1000]

# 輸出統計結果
print("total execution count:", len(score_list))
print("final score less than -1000 count:", len(scores_below_threshold))
print("score less than -1000:")
for score in scores_below_threshold:
    print(score)
