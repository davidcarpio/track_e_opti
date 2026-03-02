import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
# Use relative path if valid, or fallback to absolute
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.append(src_path)
print(f"Adding src path: {src_path}")
from track_analysis import Track, TrackPoint

class TestGradeCalculation(unittest.TestCase):
    def setUp(self):
        # Create a dummy CSV file
        self.csv_path = Path("test_track.csv")
        data = {
            'Distance from Lap Line (m)': [0, 10, 20, 30, 40],
            'Elevation (m)': [0, 1, 2, 5, 5],
            'UTMX': [0, 10, 20, 30, 40],
            'UTMY': [0, 0, 0, 0, 0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        if self.csv_path.exists():
            self.csv_path.unlink()

    def test_grade_calculation(self):
        track = Track(str(self.csv_path))
        
        # Expected grades
        # i=0: (1-0)/(10-0) = 0.1
        # i=1: (2-1)/(20-10) NO, code uses (i+1 - i-1) => (2-0)/(20-0) = 0.1
        # i=2: (5-1)/(30-10) = 4/20 = 0.2
        # i=3: (5-2)/(40-20) = 3/20 = 0.15
        # i=4: (5-5)/(40-30) = 0.0
        
        expected_grades = [0.1, 0.1, 0.2, 0.15, 0.0]
        
        calculated_grades = [p.grade for p in track.points]
        
        print("Calculated grades:", calculated_grades)
        print("Expected grades:", expected_grades)
        
        np.testing.assert_allclose(calculated_grades, expected_grades, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
