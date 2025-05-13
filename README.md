# Shape Detector

An intelligent shape detection, color recognition, and measurement application with advanced wood tone detection.

## Features

- **Shape Detection**: Accurately identifies various shapes (circle, square, rectangle, triangle, etc.)
- **Color Recognition**: Enhanced color detection with special handling for wood tones, natural colors, and accurate naming of red/pink shades
- **Size Measurement**: Calculates dimensions of detected shapes
- **Combined Detection**: Provides shape, color, and size information in a single analysis

## Project Setup

### Prerequisites

- Python 3.6 or newer
- Git
- Node.js and npm (for the frontend)

### Cloning the Repository

```bash
git clone https://github.com/YOUR_USERNAME/shape-detector.git
cd shape-detector
```

### Backend Setup

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
python app.py
```

The backend server will start at http://localhost:5000

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the frontend development server
npm start
```

The frontend will be available at http://localhost:3000

## Color Detection System

The project includes an enhanced color detection system with special handling for natural tones like wood and accurate naming for various color spectrums:

- Uses HSV color space for better color matching
- Special handling for wood tones and browns
- Intelligent filtering to avoid shadows and reflections
- Prioritizes brighter, more representative colors
- Improved color naming that correctly identifies red, pink, and crimson shades
- Direct RGB range mapping for common colors to ensure intuitive naming

### Testing Color Detection

You can test the color detection system using the provided test scripts:

```bash
cd backend
python color_test.py       # Test general color detection
python wood_test.py        # Test wood tone detection specifically
python color_naming_test.py # Test the color naming accuracy
```

The `color_naming_test.py` script generates a visual reference with color swatches and their detected names, making it easy to verify correct color identification.

## API Endpoints

- `/upload`: Upload an image
- `/detect-shape`: Detect shapes in the image
- `/detect-color`: Detect colors in the image
- `/detect-size`: Calculate size of shapes in the image
- `/detect-shape-color`: Combined shape and color detection

## License

[Specify your license here]

## Credits

[Your name and any acknowledgements]
