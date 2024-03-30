
# Flask Image Generation Application

This is a simple Flask application designed to upload and display images with an option to apply machine learning models for image generation. It consists of a home page, an upload page, and a display page where the generated image can be viewed.

## Getting Started

These instructions will get your copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

What things you need to install the software and how to install them:

```bash
Python 3.6 or higher
Flask
```

### Installing

A step by step series of examples that tell you how to get a development environment running:

1. Clone the repository to your local machine:

```bash
git clone <https://yourrepositorylink.com>
```

2. Navigate to the project directory:

```bash
cd path_to_your_project
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment:

On Windows:

```bash
.\venv\Scripts\activate
```

On Unix or MacOS:

```bash
source venv/bin/activate
```

5. Install the required packages:

```bash
pip install -r requirements.txt
``` 

6. Set environment variables:

On Windows:

```cmd
set FLASK_APP=app.py
set FLASK_ENV=development
```

On Unix or MacOS:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
```

7. Run the Flask application:

```bash
flask run
```

Your application should now be running on http://127.0.0.1:5000.

## Usage

- Visit http://127.0.0.1:5000 to view the home page.
- Click on "Get Started" to navigate to the image upload page.
- Upload an image and wait for the processing to complete.
- View the generated image on the display page.