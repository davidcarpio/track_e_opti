"""
Track Analysis for Shell Eco-marathon

This module processes track data to extract:
- Road curvature at each point
- Road grade (slope) from elevation
- Identification of straights and corners
- Stop location recommendations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class TrackPoint:
    """Single point on the track."""
    distance: float  # Distance from lap line in m
    elevation: float  # Elevation in m
    x: float  # UTM X coordinate
    y: float  # UTM Y coordinate
    curvature: float = 0.0  # 1/radius in 1/m
    grade: float = 0.0  # Rise/run
    max_velocity: float = 40.0 / 3.6  # Max cornering velocity in m/s


@dataclass
class TrackSegment:
    """Segment of track (straight or corner)."""
    start_distance: float
    end_distance: float
    segment_type: str  # 'straight' or 'corner'
    avg_curvature: float
    avg_grade: float
    min_radius: float


class Track:
    """Track analysis and data container."""
    
    def __init__(self, csv_path: str):
        """
        Load and process track data from CSV.
        
        Args:
            csv_path: Path to track CSV file
        """
        self.csv_path = Path(csv_path)
        self.points: List[TrackPoint] = []
        self.segments: List[TrackSegment] = []
        self.total_distance: float = 0.0
        
        self._load_data()
        self._compute_curvature()
        self._compute_grade()
        self._identify_segments()
    
    def _load_data(self):
        """Load track data from CSV."""
        df = pd.read_csv(self.csv_path)
        
        # Handle column names with BOM or spaces
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        for _, row in df.iterrows():
            point = TrackPoint(
                distance=row['Distance from Lap Line (m)'],
                elevation=row['Elevation (m)'],
                x=row['UTMX'],
                y=row['UTMY']
            )
            self.points.append(point)
        
        self.total_distance = self.points[-1].distance
        print(f"Loaded {len(self.points)} track points, total distance: {self.total_distance:.1f} m")
    
    def _compute_curvature(self, window: int = 5):
        """
        Compute curvature at each point using local circle fitting.
        
        Curvature = 1/R where R is the radius of the osculating circle.
        Uses three-point circle fitting with smoothing window.
        """
        n = len(self.points)
        
        for i in range(n):
            # Get points for curvature calculation
            i_prev = max(0, i - window)
            i_next = min(n - 1, i + window)
            
            if i_next - i_prev < 2:
                self.points[i].curvature = 0.0
                continue
            
            # Use three points for circle fitting
            p1 = self.points[i_prev]
            p2 = self.points[i]
            p3 = self.points[i_next]
            
            # Menger curvature formula: 4*Area / (|P1-P2| * |P2-P3| * |P3-P1|)
            x1, y1 = p1.x, p1.y
            x2, y2 = p2.x, p2.y
            x3, y3 = p3.x, p3.y
            
            # Signed area of triangle * 2
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            
            # Edge lengths
            d12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            d23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
            d31 = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
            
            denom = d12 * d23 * d31
            if denom > 1e-10:
                curvature = 2 * area2 / denom
            else:
                curvature = 0.0
            
            self.points[i].curvature = curvature
        
        # Smooth curvature
        self._smooth_curvature(window=3)
    
    def _smooth_curvature(self, window: int = 3):
        """Apply moving average smoothing to curvature."""
        curvatures = np.array([p.curvature for p in self.points])
        
        # Simple moving average
        kernel = np.ones(2 * window + 1) / (2 * window + 1)
        smoothed = np.convolve(curvatures, kernel, mode='same')
        
        for i, p in enumerate(self.points):
            p.curvature = smoothed[i]
    
    def _compute_grade(self):
        """Compute road grade (slope) at each point."""
        n = len(self.points)
        
        for i in range(n):
            if i == 0:
                de = self.points[1].elevation - self.points[0].elevation
                ds = self.points[1].distance - self.points[0].distance
            elif i == n - 1:
                de = self.points[i].elevation - self.points[i-1].elevation
                ds = self.points[i].distance - self.points[i-1].distance
            else:
                de = self.points[i+1].elevation - self.points[i-1].elevation
                ds = self.points[i+1].distance - self.points[i-1].distance
            
            if ds > 0:
                self.points[i].grade = de / ds
            else:
                self.points[i].grade = 0.0
    
    def _identify_segments(self, curvature_threshold: float = 0.01):
        """
        Identify straights and corners based on curvature.
        
        Args:
            curvature_threshold: Curvature above this is a corner (1/m)
        """
        self.segments = []
        
        if not self.points:
            return
        
        # Classify each point
        in_corner = abs(self.points[0].curvature) > curvature_threshold
        segment_start = 0
        
        for i, p in enumerate(self.points):
            is_corner = abs(p.curvature) > curvature_threshold
            
            if is_corner != in_corner or i == len(self.points) - 1:
                # End current segment
                segment_points = self.points[segment_start:i]
                if segment_points:
                    avg_curv = np.mean([abs(sp.curvature) for sp in segment_points])
                    avg_grade = np.mean([sp.grade for sp in segment_points])
                    min_radius = 1.0 / max(avg_curv, 0.001)
                    
                    segment = TrackSegment(
                        start_distance=self.points[segment_start].distance,
                        end_distance=self.points[i-1].distance if i > 0 else self.points[0].distance,
                        segment_type='corner' if in_corner else 'straight',
                        avg_curvature=avg_curv,
                        avg_grade=avg_grade,
                        min_radius=min_radius
                    )
                    self.segments.append(segment)
                
                segment_start = i
                in_corner = is_corner
        
        print(f"Identified {len(self.segments)} segments: "
              f"{sum(1 for s in self.segments if s.segment_type == 'straight')} straights, "
              f"{sum(1 for s in self.segments if s.segment_type == 'corner')} corners")
    
    def get_point_at_distance(self, distance: float) -> TrackPoint:
        """
        Get track point at a given distance (interpolated if needed).
        
        Args:
            distance: Distance from lap line in m
            
        Returns:
            TrackPoint at that distance
        """
        # Handle wraparound
        distance = distance % self.total_distance
        
        # Find bracketing points
        for i in range(len(self.points) - 1):
            if self.points[i].distance <= distance <= self.points[i+1].distance:
                # Linear interpolation
                p1, p2 = self.points[i], self.points[i+1]
                t = (distance - p1.distance) / (p2.distance - p1.distance + 1e-10)
                
                return TrackPoint(
                    distance=distance,
                    elevation=p1.elevation + t * (p2.elevation - p1.elevation),
                    x=p1.x + t * (p2.x - p1.x),
                    y=p1.y + t * (p2.y - p1.y),
                    curvature=p1.curvature + t * (p2.curvature - p1.curvature),
                    grade=p1.grade + t * (p2.grade - p1.grade)
                )
        
        return self.points[-1]
    
    def get_curvature_at_distance(self, distance: float) -> float:
        """Get curvature at a given distance."""
        return self.get_point_at_distance(distance).curvature
    
    def get_grade_at_distance(self, distance: float) -> float:
        """Get grade at a given distance."""
        return self.get_point_at_distance(distance).grade
    
    def get_radius_at_distance(self, distance: float) -> float:
        """Get corner radius at a given distance."""
        curv = abs(self.get_curvature_at_distance(distance))
        if curv < 1e-6:
            return float('inf')
        return 1.0 / curv
    
    def find_main_straight(self) -> Tuple[float, float]:
        """
        Find the longest straight section.
        
        Returns:
            (start_distance, end_distance) of main straight
        """
        straights = [s for s in self.segments if s.segment_type == 'straight']
        if not straights:
            return (0, 100)
        
        longest = max(straights, key=lambda s: s.end_distance - s.start_distance)
        return (longest.start_distance, longest.end_distance)
    
    def find_tightest_corner_sequence(self) -> Tuple[float, float]:
        """
        Find the sequence of tight corners.
        
        Returns:
            (start_distance, end_distance) before tight corner sequence
        """
        corners = [s for s in self.segments if s.segment_type == 'corner']
        if not corners:
            return (self.total_distance / 2, self.total_distance / 2 + 50)
        
        # Find corner with smallest radius
        tightest = min(corners, key=lambda s: s.min_radius)
        return (tightest.start_distance, tightest.end_distance)
    
    def get_worst_case_stop_locations(self) -> List[float]:
        """
        Get the two worst-case mandatory stop locations.
        
        Returns:
            List of two distances for stops
        """
        # Stop 1: End of main straight (max KE)
        straight_start, straight_end = self.find_main_straight()
        stop1 = straight_end
        
        # Stop 2: Before tight corner sequence
        corner_start, corner_end = self.find_tightest_corner_sequence()
        stop2 = corner_start
        
        print(f"Worst-case stop locations:")
        print(f"  Stop 1 (end of straight): {stop1:.1f} m")
        print(f"  Stop 2 (before tight corners): {stop2:.1f} m")
        
        return [stop1, stop2]
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get track data as numpy arrays for optimization.
        
        Returns:
            (distances, elevations, curvatures, grades)
        """
        distances = np.array([p.distance for p in self.points])
        elevations = np.array([p.elevation for p in self.points])
        curvatures = np.array([p.curvature for p in self.points])
        grades = np.array([p.grade for p in self.points])
        
        return distances, elevations, curvatures, grades
    
    def summary(self) -> str:
        """Generate track summary."""
        elevations = [p.elevation for p in self.points]
        curvatures = [abs(p.curvature) for p in self.points]
        grades = [p.grade for p in self.points]
        
        return f"""
Track Summary:
  Total distance: {self.total_distance:.1f} m
  Elevation range: {min(elevations):.1f} - {max(elevations):.1f} m
  Max grade: {max(grades)*100:.1f}% uphill, {min(grades)*100:.1f}% downhill
  Max curvature: {max(curvatures):.4f} 1/m (radius = {1/max(max(curvatures), 0.001):.1f} m)
  Segments: {len(self.segments)} ({sum(1 for s in self.segments if s.segment_type == 'straight')} straights)
"""


def analyze_track(csv_path: str) -> Track:
    """
    Load and analyze track from CSV file.
    
    Args:
        csv_path: Path to track CSV
        
    Returns:
        Track object with full analysis
    """
    track = Track(csv_path)
    print(track.summary())
    return track


if __name__ == "__main__":
    # Test with actual track data
    _project_root = Path(__file__).resolve().parent.parent
    track = analyze_track(str(_project_root / "data" / "tracks" / "sem_2025_eu.csv"))
    
    # Find stop locations
    stops = track.get_worst_case_stop_locations()
    
    # Plot track overview
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    distances, elevations, curvatures, grades = track.get_arrays()
    
    # Elevation profile
    axes[0].plot(distances, elevations, 'b-')
    axes[0].set_ylabel('Elevation (m)')
    axes[0].set_title('Track Profile')
    axes[0].axvline(stops[0], color='r', linestyle='--', label='Stop 1')
    axes[0].axvline(stops[1], color='g', linestyle='--', label='Stop 2')
    axes[0].legend()
    axes[0].grid(True)
    
    # Curvature profile
    axes[1].plot(distances, curvatures, 'r-')
    axes[1].set_ylabel('Curvature (1/m)')
    axes[1].axvline(stops[0], color='r', linestyle='--')
    axes[1].axvline(stops[1], color='g', linestyle='--')
    axes[1].grid(True)
    
    # Grade profile
    axes[2].plot(distances, grades * 100, 'g-')
    axes[2].set_xlabel('Distance (m)')
    axes[2].set_ylabel('Grade (%)')
    axes[2].axvline(stops[0], color='r', linestyle='--')
    axes[2].axvline(stops[1], color='g', linestyle='--')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(str(_project_root / 'results' / 'track_analysis.png'), dpi=150)
    print("Saved track analysis to track_analysis.png")
    plt.show()
