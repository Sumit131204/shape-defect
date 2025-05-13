   # Shape Detector Application
   
   This application detects shapes in images and provides measurements of their dimensions.
   
   ## Requirements
   - Python 3.7+ 
   - Node.js and npm (for the frontend)
   
   ## Setup Instructions
   
   ### Backend Setup
   1. Navigate to the backend directory:
      ```
      cd shape-detector/backend
      ```
   
   2. Create a virtual environment (recommended):
      ```
      python -m venv venv
      ```
   
   3. Activate the virtual environment:
      - Windows: `venv\Scripts\activate`
      - Mac/Linux: `source venv/bin/activate`
   
   4. Install required packages:
      ```
      pip install -r requirements.txt
      ```
   
   5. Run the backend server:
      ```
      python app.py
      ```
      The server will start on http://localhost:5000
   
   ### Frontend Setup
   1. Navigate to the frontend directory:
      ```
      cd shape-detector/frontend
      ```
   
   2. Install dependencies:
      ```
      npm install
      ```
   
   3. Start the development server:
      ```
      npm start
      ```
      The application will open in your browser at http://localhost:3000
   
   ## Usage
   1. Upload an image containing shapes
   2. The application will detect shapes and provide measurements
   3. View the results including dimensions and area calculations