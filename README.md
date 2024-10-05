# Face Attendance System
Yeh Face Attendance System project face detection aur recognition ka istemal karke attendance log karne ke liye design kiya gaya hai. Is project mein Python, OpenCV, aur MySQL ka istemal kiya gaya hai.

## Project Ki Structure
#### 1. Face Collection Code

- ```File: Dataset.py```
- Function: Yeh code webcam se live images capture karke user ke face samples collect karta hai. User se naam aur roll number pucha jata hai, aur yeh samples specific folder mein save hote hain.


````
# Code ka Summary
face_classifier = cv2.CascadeClassifier('path_to_haarcascade_frontalface_default.xml')
````
- Key Features:
  - User se naam aur roll number lena.
  - Face detection aur cropping.
  - 100 face samples collect karna.
    
#### 2. Model Training Code

- ```File: Train.py```
- Function: Yeh code collected face samples ka istemal karke face recognition model train karta hai aur trained model ko save karta hai.
  
```
# Code ka Summary
for person_folder in listdir(data_path):
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
```
- Key Features:
 - Collected images ko load karna.
 - Labels assign karna.
 - LBPH (Local Binary Patterns Histograms) model ko train karna aur save karna.
 - 

#### 3. Attendance Logging Code

- ```File: Detection.py```
- Function: Yeh code live video stream se face detect karta hai, recognition karta hai, aur agar recognized person present hai toh attendance log karta hai.

```
# Code ka Summary
current_date = datetime.now().strftime("%Y_%m_%d")
```
- Key Features:
  - Attendance check karna aur log karna.
  - Database mein attendance entries create karna.
  - Attendance ka table dynamically create karna agar wo pehle se nahi hai.
 
  
## Database Setup
1. MySQL Server Install Karein: MySQL server install karne ke liye official website se installer download karein.
2. Database Aur Table Banayein:
   - Database banane ke liye:
```
CREATE DATABASE college;

```
3. automatic date ke hisab se Attendance table banan jayegi :

## Attendance Table Ko Dynamic Tarike Se Create Karna: Attendance logging code har din ke liye attendance table automatically create karega.

- Dependencies
- OpenCV
- NumPy
- pymysql
- Usage

### 1. Face Collection:

   - ```Dataset.py ```ko run karein.
   - Instructions follow karein aur samples collect karein.
### 2. Model Training:

   - ```Train.py```ko run karein.
   - Model train hone ke baad save hoga.
### 3. Attendance Logging:

   - ```Detection.py``` ko run karein.
   - Webcam se attendance log karne ke liye ready rahein.
     
Conclusion
Yeh project face attendance system ko asan aur effective banata hai. Aap is project ko customize bhi kar sakte hain apne needs ke according.


