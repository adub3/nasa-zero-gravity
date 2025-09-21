import os, glob

# 🔧 1) Put the path you think is correct here (try absolute to end the debate)
CSV_IN = r"C:\Users\zirun\code\Zero Gravity\nasa-zero-gravity\src\data\trainingdat\output_fullid\mergeddata.csv"

print("cwd:", os.getcwd())
print("CSV_IN:", CSV_IN)
print("exists?", os.path.exists(CSV_IN))

# 🔎 If exists=False, show me where the file actually is from Python’s POV:
hits = glob.glob("**/mergeddata.csv", recursive=True)
print("found", len(hits), "matches (showing first 10):")
for h in hits[:10]:
    print(" -", os.path.abspath(h))

# Also check if a local 'output_fullid' folder is visible from cwd:
print("output_fullid exists in cwd?", os.path.isdir("output_fullid"))
if os.path.isdir("output_fullid"):
    print("contents of ./output_fullid:", os.listdir("output_fullid")[:20])
