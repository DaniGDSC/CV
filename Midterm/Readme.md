# Computer Vision Project: Traditional Image Processing

## Overview
This project implements traditional image processing techniques for filtering, 3D reconstruction, and image stitching as required for the INS3155 Computer Vision midterm project.

## Project Details
- **Course**: INS3155 - Computer Vision
- **Release Date**: April 20, 2025
- **Submission Deadline**: May 03, 2025

## Project Objectives
Implementation and comparison of classical computer vision techniques for:
1. Image filtering and enhancement
2. 3D reconstruction from stereo images
3. Image stitching from multiple views

## Requirements

### General Requirements
- This is an individual project. Collaboration is not allowed.
- Only traditional image processing methods are permitted.
- Deep learning models or pretrained networks are not allowed.
- Allowed libraries: OpenCV, NumPy, Matplotlib, Open3D, and other non-learning libraries.

### Deliverables
- PDF report with introduction, methods, results, analysis, and conclusion
- Source code (in a ZIP file or GitHub link)
- Visualization results

### Part A: Image Filtering (20 points)
Implement and compare different traditional filters:
- Mean filter
- Gaussian filter
- Median filter
- Laplacian sharpening

**Deliverables:**
- Original, noisy, and filtered images (side-by-side comparison)
- Analysis of filter performance for noise reduction and edge preservation

### Part B: 3D Reconstruction (25 points)
Using stereo images:
- Compute disparity map using block matching or SGBM
- Reconstruct 3D point cloud from disparity
- Estimate the fundamental matrix and draw epipolar lines

**Deliverables:**
- Disparity map visualization
- 3D point cloud visualization
- Epipolar line drawings on input images
- Brief explanation of the implemented stereo algorithm

### Part C: Image Stitching (25 points)
Create an image stitching pipeline:
- Detect features using SIFT, SURF, or ORB
- Match features between four overlapping images
- Estimate homography matrix using RANSAC
- Warp and blend images into a panorama

**Deliverables:**
- Matched keypoints visualization
- Final panorama image
- Short explanation of homography estimation process

### Comparative Analysis (15 points)
For each part (A-C), compare at least two alternative methods or parameter settings:
- Discuss strengths and weaknesses
- Provide quantitative evaluation (e.g., PSNR, number of inliers)
- Include qualitative visual inspection

### Report and Code Quality (15 points)
- Well-structured report with all required sections
- Clean, documented, and organized code
- Clear visualization of results

## Grading Rubric (100 Points)
| Component | Points |
|-----------|--------|
| Image Filtering (Part A) | 20 |
| 3D Reconstruction (Part B) | 25 |
| Image Stitching (Part C) | 25 |
| Comparative Analysis | 15 |
| Report Quality | 10 |
| Code Quality and Results | 5 |

**Note**: Bonus points may be awarded for use of your own images, insightful visualizations, or additional comparative experiments.

## Project Structure
```
project/
├── data/
│   ├── input_images/
│   └── output_results/
├── src/
│   ├── filtering.py
│   ├── stereo.py
│   ├── stitching.py
│   └── utils.py
├── notebooks/
│   ├── part_a_filtering.ipynb
│   ├── part_b_stereo.ipynb
│   └── part_c_stitching.ipynb
├── main.py
└── README.md
```

## Setup and Usage
1. Clone this repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run individual notebook files for each part or execute `main.py`
4. Results will be saved in the `data/output_results/` directory

## Dependencies
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Open3D (for 3D point cloud visualization)
