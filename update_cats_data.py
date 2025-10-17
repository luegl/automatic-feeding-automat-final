"""
Automatic Ration Reset
----------------------
This program resets the remaining food rations ("ration_left")
for all cats stored in a JSON file. It runs continuously and 
resets all rations at fixed time intervals (currently every 12 hours).
"""

# Standard library
import json
import time
from datetime import datetime



# -------------------------------------------------------------------
#   Configuration
# -------------------------------------------------------------------

# Path to the JSON file storing cat data
FILE_PATH = "cats.json"



# -------------------------------------------------------------------
#   Functions
# -------------------------------------------------------------------

def reset_rations() -> None:
    """
    Reset the remaining ration ('ration_left') for each cat to its total ration ('ration_total').
    """
    # Read JSON file
    with open(FILE_PATH, "r", encoding="utf-8") as file:
        cats_data = json.load(file)

    # Update each cat’s remaining ration
    for cat, info in cats_data.items():
        info["ration_left"] = info["ration_total"]

    # Save updated data
    with open(FILE_PATH, "w", encoding="utf-8") as file:
        json.dump(cats_data, file, indent=2, ensure_ascii=False)

    # Print timestamped log
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Rations have been reset.")


def main() -> None:
    """
    Run the reset function in an infinite loop with a fixed delay.
    """
    while True:
        reset_rations()
        # Wait 12 hours (12 * 60 * 60 seconds)
        time.sleep(12 * 60 * 60)



# -------------------------------------------------------------------
#   Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
