# 🎯 People Tracker Pro (ClassPulse Analytics)

People Tracker Pro is a real-time computer vision and movement analytics dashboard designed to detect, track, and analyze human traffic using live camera feeds or RTSP streams. Built with a robust combination of Deep Learning, Statistical Estimation, and Cloud Infrastructure, this system provides highly accurate and secure actionable data.

---

## 🚀 Key Features

* **AI Person Detection:** Utilizes **YOLOv8** for rapid and accurate frame-by-frame person detection.
* **Advanced Multi-Object Tracking:** Implements a multi-object centroid tracker upgraded with a **2D Constant Velocity Kalman Filter** to smooth noisy sensor jitter and maintain target paths during temporary occlusions.
* **Visual Analytics Layer:** Features custom visual overlays including persistent movement trails with contrast stacking and motion density heatmaps.
* **Dwell-Time Tracking:** Tracks exactly how long an individual spends in a frame, logging entry and exit events with a granularity of seconds.
* **AWS Cloud Integration:** Seamlessly records sessions to local storage and automatically uploads MP4 videos and timestamped CSV data reports to **Amazon S3** (`eu-north-1`) via secure `boto3` client integrations.
* **Admin Security Protection:** Includes a persistent dashboard lock requiring authorization to adjust camera settings, detection thresholds, or simulate environment factors.

---

## 🛠️ Tech Stack & Technologies

* **Core Language:** Python 3.10+
* **Computer Vision & AI:** Ultralytics YOLOv8, OpenCV
* **Mathematical Operations:** NumPy, SciPy (Spatial Distance Estimations)
* **GUI Framework:** PyQt5 (Modern Custom Dark Theme Architecture)
* **Cloud Infrastructure:** Amazon Web Services (AWS S3, Boto3 SDK)

---

📦 Prerequisites & Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/MahmoudAlaaElhdad/People-Tracker.git](https://github.com/MahmoudAlaaElhdad/People-Tracker.git)
   cd People-Tracker
Install Dependencies:

Bash
pip install -r requirements.txt
Configure AWS Credentials:
Ensure you have your AWS CLI configured properly on your host machine to allow boto3 to securely upload files without hardcoding credentials:

Bash
aws configure
Run the Application:

Bash
python main11.py
🔒 Security Disclaimer
This project handles AWS infrastructure connectivity natively through standard secure credential files or environmental variables. All private production access keys have been omitted from the repository codebase to comply with Cloud DevSecOps best practices.
